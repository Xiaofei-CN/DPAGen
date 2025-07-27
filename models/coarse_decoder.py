from einops import rearrange
import torch
from torch import nn
from typing import Any, Dict, Optional
from diffusers.models.attention import AdaLayerNorm, AdaLayerNormZero, FeedForward
from diffusers.models.attention_processor import *


class BasicTransBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention1_dim: Optional[int] = None,
            cross_attention2_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            triple_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention1_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attn
        if cross_attention1_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention1_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        if cross_attention2_dim is not None or double_self_attention:
            self.norm4 = (
                AdaLayerNorm(dim, num_embeds_ada_norm)
                if self.use_ada_layer_norm
                else nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            )
            self.attn3 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention2_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm4 = None
            self.attn3 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            query_pos: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_multi_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 1. Cross-Attention
        if self.attn2 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        if self.attn3 is not None:
            hidden_states = hidden_states + query_pos
            norm_hidden_states = (
                self.norm4(hidden_states, timestep) if self.use_ada_layer_norm else self.norm4(hidden_states)
            )

            attn_output = self.attn3(
                norm_hidden_states,
                encoder_hidden_states=encoder_multi_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
class TransBlockWithCrossAttention(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
    """
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention1_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",
            final_dropout: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = True

        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Cross-Attn
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif self.use_ada_layer_norm_zero:
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention1_dim,
            upcast_attention=upcast_attention,
        )

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn, final_dropout=final_dropout)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            query_pos: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
    ):
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        # 2. Self-Attention
        hidden_states = hidden_states + query_pos
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self._chunk_dim] % self._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self._chunk_dim]} has to be divisible by chunk size: {self._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self._chunk_dim] // self._chunk_size
            ff_output = torch.cat(
                [self.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self._chunk_dim)],
                dim=self._chunk_dim,
            )
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states


class CoarseDecoderType1(nn.Module):
    """
    Parameters:108 584 960
    """

    def __init__(self, n_ctx=64, ctx_dim=768, heads=24, depth=8, depths=[], pose_channel=320, pose_dim=1024,pre_seq_len=None):
        super().__init__()

        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth

        self.kv_pos_embed = nn.Parameter(torch.zeros(pose_dim, pose_channel))
        nn.init.normal_(self.kv_pos_embed, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if pre_seq_len is not None:
            self.iter_embed = nn.Parameter(torch.zeros(1, n_ctx * pre_seq_len, ctx_dim))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.iter_embed = None

        self.blocks = []
        for _ in range(depth):
            self.blocks.append(BasicTransBlock(
                dim=ctx_dim,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                cross_attention1_dim=pose_channel,
                cross_attention2_dim=ctx_dim,
            ))
        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, encoder_multi_hidden_states):
        B, L, C = hidden_states.shape
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')  # bs 1024 320
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
        encoder_hidden_states = encoder_hidden_states + kv_pos_embed
        if self.iter_embed is not None:
            iter_embed = self.iter_embed.expand(B, -1, -1)
            encoder_multi_hidden_states = encoder_multi_hidden_states + iter_embed
        pos_embed = self.pos_embed.expand(B, -1, -1)

        for blk in self.blocks:
            hidden_states = blk(hidden_states, pos_embed,
                                encoder_hidden_states=encoder_hidden_states,
                                encoder_multi_hidden_states=encoder_multi_hidden_states)
        return hidden_states


class CoarseDecoderType2(nn.Module):
    """
    Parameters:99 138 560
    there are two types of CoarseDecoder block

    the first decoder block with three attention (corss attention x 2, self attention x 1 ) (locate in 2,4,6,8)
    for the first corss attention layer, Source image feature corss attention with Target pose feature
    the second corss attention layer, first one corss attention with coarse pose feature again

    the second decoder block with two attention (corss attention x 1, self attention x 1 )
    for the first corss attention layer, Source image feature corss attention with Target pose feature
    the last self attention layer,

    """

    def __init__(self, n_ctx=257, ctx_dim=768, heads=8, depth=8, depths=[2, 4, 6, 8], pose_channel=320, pose_dim=1024,pre_seq_len=None):
        super().__init__()

        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth
        self.depths = depths

        self.kv_pos_embed = nn.Parameter(torch.zeros(pose_dim, pose_channel))
        nn.init.normal_(self.kv_pos_embed, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if pre_seq_len is not None:
            self.iter_embed = nn.Parameter(torch.zeros(1, n_ctx * pre_seq_len, ctx_dim))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.iter_embed = None

        self.blocks = []
        for i in range(1, depth + 1):
            if i in depths:
                self.blocks.append(BasicTransBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim // heads,
                    cross_attention1_dim=pose_channel,
                    cross_attention2_dim=ctx_dim,
                ))
            else:
                self.blocks.append(BasicTransBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim // heads,
                    cross_attention1_dim=pose_channel,
                ))

        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, encoder_multi_hidden_states):
        B, L, C = hidden_states.shape
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')  # bs 1024 320
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
        encoder_hidden_states = encoder_hidden_states + kv_pos_embed
        if self.iter_embed is not None:
            iter_embed = self.iter_embed.expand(B, -1, -1)
            encoder_multi_hidden_states = encoder_multi_hidden_states + iter_embed
        pos_embed = self.pos_embed.expand(B, -1, -1)

        for index, blk in enumerate(self.blocks):
            if index + 1 in self.depths:
                hidden_states = blk(hidden_states, pos_embed,
                                    encoder_hidden_states=encoder_hidden_states,
                                    encoder_multi_hidden_states=encoder_multi_hidden_states)
            else:
                hidden_states = blk(hidden_states, pos_embed,
                                    encoder_hidden_states=encoder_hidden_states, )

        return hidden_states


class CoarseDecoderType3(nn.Module):
    """
    Parameters:93 630 464
    there are two types of CoarseDecoder block

    the first decoder block with two attention (corss attention x 1, self attention x 1 ) (locate in 1,3,5,7)
    for the first corss attention layer, Source image feature corss attention with Target pose feature
    the last self attention layer,

    the second decoder block with two attention (corss attention x 1, self attention x 1 )(locate in 2,4,6,8)
    for the first corss attention layer, Source image feature corss attention with Coarse pose feature
    the last self attention layer,
    """

    def __init__(self, n_ctx=257, ctx_dim=768, heads=8, depth=8, depths=[2, 4, 6, 8], pose_channel=320, pose_dim=1024,pre_seq_len=None):
        super().__init__()

        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth
        self.depths = depths

        self.kv_pos_embed = nn.Parameter(torch.zeros(pose_dim, pose_channel))
        nn.init.normal_(self.kv_pos_embed, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if pre_seq_len is not None:
            self.iter_embed = nn.Parameter(torch.zeros(1, n_ctx * pre_seq_len, ctx_dim))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.iter_embed = None

        self.blocks = []
        for i in range(1, depth + 1):
            if i in depths:
                self.blocks.append(BasicTransBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim // heads,
                    cross_attention1_dim=ctx_dim,
                ))
            else:
                self.blocks.append(BasicTransBlock(
                    dim=ctx_dim,
                    num_attention_heads=heads,
                    attention_head_dim=ctx_dim // heads,
                    cross_attention1_dim=pose_channel,
                ))

        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, encoder_multi_hidden_states):
        B, L, C = hidden_states.shape
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')  # bs 1024 320
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)
        encoder_hidden_states = encoder_hidden_states + kv_pos_embed
        if self.iter_embed is not None:
            iter_embed = self.iter_embed.expand(B, -1, -1)
            encoder_multi_hidden = encoder_multi_hidden_states + iter_embed
        else:
            encoder_multi_hidden = encoder_multi_hidden_states
        pos_embed = self.pos_embed.expand(B, -1, -1)

        for index, blk in enumerate(self.blocks):
            if index + 1 in self.depths:
                hidden_states = blk(hidden_states, pos_embed,
                                    encoder_hidden_states=encoder_multi_hidden)
            else:
                hidden_states = blk(hidden_states, pos_embed,
                                    encoder_hidden_states=encoder_hidden_states)

        return hidden_states


class CoarseDecoderType4(nn.Module):
    """
    this model do cross attention with Tgt pose feature  and coarse pose feature
    then pipeline as old CoarseDecoder block
    """

    def __init__(self, n_ctx=257, ctx_dim=768, heads=8, depth=8, depths=[], pose_channel=320, pose_dim=1024,pre_seq_len=None):
        super().__init__()
        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth
        self.depths = depths

        self.kv_pos_embed = nn.Parameter(torch.zeros(pose_dim, pose_channel))
        nn.init.normal_(self.kv_pos_embed, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if pre_seq_len is not None:
            self.iter_embed = nn.Parameter(torch.zeros(1, n_ctx * pre_seq_len, ctx_dim))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.iter_embed = None

        self.norm = nn.LayerNorm(pose_channel, elementwise_affine=True)
        self.pose_attn = Attention(
            query_dim=pose_channel,
            heads=heads,
            dim_head=ctx_dim // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=ctx_dim,
            upcast_attention=False,
        )

        self.blocks = []
        for i in range(depth):
            self.blocks.append(BasicTransBlock(
                dim=ctx_dim,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                cross_attention1_dim=pose_channel,
            ))

        self.blocks = nn.ModuleList(self.blocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, encoder_multi_hidden_states):
        B, L, C = hidden_states.shape
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')  # bs 1024 320
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)

        if self.iter_embed is not None:
            iter_embed = self.iter_embed.expand(B, -1, -1)
            encoder_multi_hidden_states = encoder_multi_hidden_states + iter_embed
        pos_embed = self.pos_embed.expand(B, -1, -1)

        # first corss attention get k v
        encoder_hidden_states = encoder_hidden_states + kv_pos_embed
        norm_encoder_hidden_states = self.norm(encoder_hidden_states)
        cross_attention_kwargs = {}
        attn_output = self.pose_attn(
            norm_encoder_hidden_states,
            encoder_hidden_states=encoder_multi_hidden_states,
            attention_mask=None,
            **cross_attention_kwargs,
        )
        encoder_hidden_states = attn_output + encoder_hidden_states

        for blk in self.blocks:
            hidden_states = blk(hidden_states, pos_embed,
                                encoder_hidden_states=encoder_hidden_states)
        return hidden_states

class CoarseDecoderType5(nn.Module):
    """
    this model do cross attention with Tgt pose feature  and coarse pose feature
    then pipeline as old CoarseDecoder block
    """

    def __init__(self, n_ctx=257, ctx_dim=768, heads=8, depth=8, depths=[], pose_channel=320, pose_dim=1024,pre_seq_len=None):
        super().__init__()
        self.pose_channel = pose_channel
        self.ctx_dim = ctx_dim
        self.depth = depth
        self.depths = depths

        self.kv_pos_embed = nn.Parameter(torch.zeros(pose_dim, pose_channel))
        nn.init.normal_(self.kv_pos_embed, std=0.02)

        self.pos_embed = nn.Parameter(torch.zeros(1, n_ctx, ctx_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        if pre_seq_len is not None:
            self.iter_embed = nn.Parameter(torch.zeros(1, n_ctx * pre_seq_len, ctx_dim))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.iter_embed = None


        self.blocks = []
        self.preblocks = []
        for i in range(depth):
            self.blocks.append(BasicTransBlock(
                dim=ctx_dim,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                cross_attention1_dim=pose_channel,
            ))
            self.preblocks.append(TransBlockWithCrossAttention(
                dim=pose_channel,
                num_attention_heads=heads,
                attention_head_dim=ctx_dim // heads,
                cross_attention1_dim=ctx_dim,
            ))

        self.blocks = nn.ModuleList(self.blocks)
        self.preblocks = nn.ModuleList(self.preblocks)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, hidden_states, encoder_hidden_states, encoder_multi_hidden_states):
        B, L, C = hidden_states.shape
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b c h w -> b (h w) c')  # bs 1024 320
        kv_pos_embed = self.kv_pos_embed.expand(B, -1, -1)

        if self.iter_embed is not None:
            iter_embed = self.iter_embed.expand(B, -1, -1)
            encoder_multi_hidden_states = encoder_multi_hidden_states + iter_embed

        pos_embed = self.pos_embed.expand(B, -1, -1)

        # first corss attention get k v
        # encoder_hidden_states = encoder_hidden_states + kv_pos_embed

        for pblk,blk in zip(self.preblocks,self.blocks):
            encoder_hidden_states = pblk(encoder_hidden_states, kv_pos_embed,
                                         encoder_hidden_states=encoder_multi_hidden_states)
            hidden_states = blk(hidden_states, pos_embed,
                                encoder_hidden_states=encoder_hidden_states)
        return hidden_states


def CoarseDecoder(docertype, n_ctx=257, ctx_dim=768, heads=8, depth=8, depths=[], pose_channel=320, pose_dim=1024,pre_seq_len=None):
    assert docertype in ['1', '2', '3', '4', '5']
    if docertype == '1':
        return CoarseDecoderType1(
            n_ctx=n_ctx, ctx_dim=ctx_dim, heads=heads, depth=depth, depths=depths, pose_channel=pose_channel,
            pose_dim=pose_dim,pre_seq_len=pre_seq_len)
    elif docertype == '2':
        return CoarseDecoderType2(
            n_ctx=n_ctx, ctx_dim=ctx_dim, heads=heads, depth=depth, depths=depths, pose_channel=pose_channel,
            pose_dim=pose_dim,pre_seq_len=pre_seq_len)
    elif docertype == '3':
        return CoarseDecoderType3(
            n_ctx=n_ctx, ctx_dim=ctx_dim, heads=heads, depth=depth, depths=depths, pose_channel=pose_channel,
            pose_dim=pose_dim,pre_seq_len=pre_seq_len)
    elif docertype == '5':
        return CoarseDecoderType5(
            n_ctx=n_ctx, ctx_dim=ctx_dim, heads=heads, depth=depth, depths=depths, pose_channel=pose_channel,
            pose_dim=pose_dim,pre_seq_len=pre_seq_len)
    else:
        return CoarseDecoderType4(
            n_ctx=n_ctx, ctx_dim=ctx_dim, heads=heads, depth=depth, depths=depths, pose_channel=pose_channel,
            pose_dim=pose_dim,pre_seq_len=pre_seq_len)


# model = CoarseDecoderType1().cuda()
# trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
# features=torch.zeros((2,64,768)).cuda()
# pose_features = torch.zeros((2,320,32,32)).cuda()
# encoder_multi_hidden_states = torch.zeros((2,64*7,768)).cuda()
# outputs = model(features, pose_features, encoder_multi_hidden_states)
# print(outputs.shape)
