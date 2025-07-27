"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import torch
import torch.nn as nn
from diffusers.models.attention import BasicTransformerBlock


class AppearanceEncoder(nn.Module):
    def __init__(self, attn_residual_block_idx, inner_dims, ctx_dims, embed_dims, heads, depth,
                 convin_kernel_size, convin_stride, convin_padding):
        super().__init__()
        self.attn_residual_block_idx = attn_residual_block_idx
        self.inner_dims = inner_dims
        self.ctx_dims = ctx_dims
        self.embed_dims = embed_dims

        self.zero_conv_ins = []
        self.zero_conv_outs = []
        self.blocks = []
        for inner_dim, embed_dim, ctx_dim, num_head, kernel_size, stride, padding in \
            zip(inner_dims, self.embed_dims, self.ctx_dims, heads, convin_kernel_size, convin_stride, convin_padding):
            self.zero_conv_ins.append(nn.Conv2d(inner_dim, embed_dim, kernel_size=kernel_size,
                                                stride=stride, padding=padding))
            self.zero_conv_outs.append(nn.Conv2d(embed_dim, ctx_dim, kernel_size=1, stride=1, padding=0))
            self.blocks.append(nn.Sequential(*[BasicTransformerBlock(
                dim=embed_dim,
                num_attention_heads=num_head,
                attention_head_dim=embed_dim//num_head,
                double_self_attention=True
            ) for _ in range(depth)]))

        self.blocks = nn.ModuleList(self.blocks)
        self.zero_conv_ins = nn.ModuleList(self.zero_conv_ins)
        self.zero_conv_outs = nn.ModuleList(self.zero_conv_outs)

        for n in self.zero_conv_ins.parameters():
            nn.init.zeros_(n)
        for n in self.zero_conv_outs.parameters():
            nn.init.zeros_(n)

        # enable xformers
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(False, attention_op=None)
                # module.set_use_memory_efficient_attention_xformers(True, attention_op=None)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, torch.nn.Module):
                fn_recursive_set_mem_eff(module)

    def forward(self, features):
        additional_residuals = []

        for i, block in enumerate(self.blocks):
            in_H = in_W = int(features[0].shape[1] ** 0.5)
            hidden_states = features[0].permute(0, 2, 1).reshape(-1, self.inner_dims[i], in_H, in_W)
            hidden_states = self.zero_conv_ins[i](hidden_states)
            H = W = hidden_states.shape[2]
            hidden_states = hidden_states.reshape(-1, self.embed_dims[i], H * W).permute(0, 2, 1)

            hidden_states = block(hidden_states)

            hidden_states = hidden_states.permute(0, 2, 1).reshape(-1, self.embed_dims[i], H, W)
            hidden_states = self.zero_conv_outs[i](hidden_states)
            # hidden_states = hidden_states.reshape(-1, self.ctx_dims[i], H * W).permute(0, 2, 1)
            additional_residuals.append(hidden_states)
            if i != len(self.blocks) - 1 and self.inner_dims[i] != self.inner_dims[i + 1]:
                features.pop(0)
        return additional_residuals