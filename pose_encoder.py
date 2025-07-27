"""
@author: Yanzuo Lu
@author: oliveryanzuolu@gmail.com
"""
import torch
import torch.nn as nn
from diffusers.models.resnet import ResnetBlock2D, Downsample2D
from diffusers.utils import BaseOutput

class PoseEncoderOutPuts(BaseOutput):
    pose_features: torch.Tensor = None
    img_features: torch.Tensor = None

class PoseEncoder(nn.Module):
    def __init__(self,pose_channels=3, channels=[64,128,320,512,768]):
        super().__init__()
        in_channels = channels[0]
        self.conv_in = nn.Conv2d(pose_channels,in_channels,kernel_size=1)

        resnets = []
        downsamplers = []
        for i in range(len(channels)):
            in_channels = in_channels if i==0 else channels[i-1]
            out_channels = channels[i]

            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=None, # no time embed
            ))
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                out_channels=out_channels,
                padding=1,
                name="op"
            ))

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        feature = []
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = downsampler(hidden_states)
            feature.append(hidden_states)
        return PoseEncoderOutPuts(pose_features=feature[2],img_features=feature[-1])
class OnlyPoseEncoderOutPuts(BaseOutput):
    pose_features: torch.Tensor = None
    img_features: torch.Tensor = None

class OnlyPoseEncoder(nn.Module):
    def __init__(self,pose_channels=3, channels=[64,128,320,512,768]):
        super().__init__()
        in_channels = channels[0]
        self.conv_in = nn.Conv2d(pose_channels,in_channels,kernel_size=1)

        resnets = []
        downsamplers = []
        for i in range(len(channels)):
            in_channels = in_channels if i==0 else channels[i-1]
            out_channels = channels[i]

            resnets.append(ResnetBlock2D(
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=None, # no time embed
            ))
            downsamplers.append(Downsample2D(
                out_channels,
                use_conv=False,
                out_channels=out_channels,
                padding=1,
                name="op"
            ))

        self.resnets = nn.ModuleList(resnets)
        self.downsamplers = nn.ModuleList(downsamplers)

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)
        for resnet, downsampler in zip(self.resnets, self.downsamplers):
            hidden_states = resnet(hidden_states, temb=None)
            hidden_states = downsampler(hidden_states)
        return hidden_states
