from collections import OrderedDict
import torch.nn.functional as F
from torch import nn
from utils import ConvNormActBlock


class FeaturePyramidNetwork(nn.Module):
    def __init__(
        self,
        input_keys_list,
        input_channels_list,
        output_channels,
        pool=True,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()
        self.input_keys_list = input_keys_list
        self.input_channels_list = input_channels_list
        self.out_channels = output_channels
        self.pool = pool

        self.channel_projections = nn.ModuleList()
        self.blend_convs = nn.ModuleList()
        for in_channels in input_channels_list:
            self.channel_projections.append(
                ConvNormActBlock(
                    in_channels, output_channels, kernel_size=1, norm_layer=norm_layer
                )
            )
            self.blend_convs.append(
                ConvNormActBlock(
                    output_channels, output_channels, kernel_size=3, norm_layer=norm_layer
                )
            )

    def forward(self, x):
        out = dict()

        res2 = self.channel_projections[0](x["res2"])
        res3 = self.channel_projections[1](x["res3"])
        res4 = self.channel_projections[2](x["res4"])
        res5 = self.channel_projections[3](x["res5"])

        fpn5 = self.blend_convs[3](res5)
        fpn4 = self.blend_convs[2](res4 + F.interpolate(res5, size=res4.shape[-2:], mode="nearest"))
        fpn3 = self.blend_convs[1](res3 + F.interpolate(res4 + F.interpolate(res5, size=res4.shape[-2:], mode="nearest"), size=res3.shape[-2:], mode="nearest"))
        fpn2 = self.blend_convs[0](res2 + F.interpolate(res3 + F.interpolate(res4 + F.interpolate(res5, size=res4.shape[-2:], mode="nearest"), size=res3.shape[-2:], mode="nearest"), size=res2.shape[-2:], mode="nearest"))

        out["fpn5"] = fpn5
        out["fpn4"] = fpn4
        out["fpn3"] = fpn3
        out["fpn2"] = fpn2

        out["fpn_pool"] = F.max_pool2d(fpn5, kernel_size=1, stride=2)

        ordered_out = OrderedDict()
        for k in ["fpn2", "fpn3", "fpn4", "fpn5", "fpn_pool"]:
            ordered_out[k] = out[k]

        return ordered_out



