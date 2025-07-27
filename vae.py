"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""

import diffusers
import torch
import torch.nn as nn
import os



class VariationalAutoencoder(nn.Module):
    def __init__(self, pretrained_path,finetune_path=None):
        super().__init__()
        self.model = diffusers.AutoencoderKL.from_pretrained(pretrained_path, use_safetensors=True)
        if finetune_path is not None:
            self.model.load_state_dict(torch.load(
                os.path.join(finetune_path, "pytorch_model.bin"), map_location="cpu"
                ), strict=False)
        self.model.requires_grad_(False)
        self.model.enable_slicing()

    def forward(self, x):
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        z = 1. / self.model.scaling_factor * z
        x = self.model.decode(z).sample
        return x

    @torch.no_grad()
    def encode(self, x):
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        return z

    @torch.no_grad()
    def decode(self, z):
        z = 1. / self.model.scaling_factor * z
        x = self.model.decode(z).sample
        return x

    def decodec(self, z):
        bs = z.shape[0]# bs, 24, 32, 32
        z = torch.chunk(z,chunks=6,dim=1)
        z = torch.stack(z, dim=1)
        z = z.reshape(-1,*z.shape[-3:])
        z = 1. / self.model.scaling_factor * z
        x = self.model.decode(z).sample
        x = x.reshape(bs,-1,3,256,256)
        return x

    @torch.no_grad()
    def encodec(self, x,y):
        bs = x.shape[0]
        x = torch.cat([x,y.reshape(-1,3,256,256)])
        z = self.model.encode(x).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z

        zx = z[:bs].repeat(1,6,1,1)
        zy = z[bs:].reshape(bs,-1,32,32)
        z = torch.cat([zx,zy])
        """bs = y.shape[0]
        z = self.model.encode(y.reshape(-1,3,256,256)).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        z = z.reshape(bs,-1,32,32)"""
        return z
    @torch.no_grad()
    def encodeimgcatpose(self, x,y=None,xp=None,yp=None):
        bs = x.shape[0]
        if y is not None:
            x = torch.cat([x.reshape(-1, 3, 256, 256), y.reshape(-1, 3, 256, 256)])
            z = self.model.encode(x).latent_dist
            z = z.sample()
            z = self.model.scaling_factor * z  # bs*6+bs*6 4, 32,32
        else:
            z = self.model.encode(x.reshape(-1, 3, 256, 256)).latent_dist
            z = z.sample()
            z = self.model.scaling_factor * z
            z = z.reshape(bs, -1,4, 32, 32)



        # if xp is not None and yp is not None:
        #     xp = torch.cat([xp.reshape(-1, 3, 256, 256), yp.reshape(-1, 3, 256, 256)])
        #     zp = self.model.encode(xp).latent_dist
        #     zp = zp.sample()
        #     zp = self.model.scaling_factor * zp  # bs*6+bs*6 4, 32,32
        #     z = torch.cat([z, zp], dim=1) # bs*6+bs*6 8, 32,32
        #
        if y is not None:
            zx = z[:bs * 6].reshape(bs, -1, 32, 32)  ## bs,48, 32,32
            zy = z[bs * 6:].reshape(bs, -1, 32, 32)
            z = torch.cat([zx, zy])

        """bs = y.shape[0]
        z = self.model.encode(y.reshape(-1,3,256,256)).latent_dist
        z = z.sample()
        z = self.model.scaling_factor * z
        z = z.reshape(bs,-1,32,32)"""
        return z
