"""Stolen from https://github.com/archinetai/a-unet readme."""

from turtle import forward
from typing import Sequence, Optional, Callable

import einops
import torch
from torch import Tensor
from a_unet import TimeConditioningPlugin, ClassifierFreeGuidancePlugin
from a_unet.apex import (
    XUNet,
    XBlock,
    ResnetItem as R,
    AttentionItem as A,
    CrossAttentionItem as C,
    ModulationItem as M,
    SkipCat
)

class UNet(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        channels: Sequence[int],
        factors: Sequence[int],
        items: Sequence[int],
        attentions: Sequence[int],
        # cross_attentions: Sequence[int],
        attention_features: int,
        attention_heads: int,
        # embedding_features: Optional[int] = None,
        skip_t: Callable = SkipCat,
        resnet_groups: int = 8,
        modulation_features: int = 1024,
        # embedding_max_length: int = 0,
        # use_classifier_free_guidance: bool = False,
        out_channels: Optional[int] = None,
        token_size: int | None = None,
    ) -> None:
        super().__init__()
        self.token_size = token_size if token_size is not None else 1
        # Check lengths
        num_layers = len(channels)
        sequences = (channels, factors, items, attentions)
        assert all(len(sequence) == num_layers for sequence in sequences)

        # Define UNet type with time conditioning and CFG plugins
        UNet = TimeConditioningPlugin(XUNet)
        # if use_classifier_free_guidance:
        #     UNet = ClassifierFreeGuidancePlugin(UNet, embedding_max_length)

        self._net = UNet(
            dim=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            blocks=[
                XBlock(
                    channels=channels,
                    factor=factor,
                    items=([R, M] + [A] * n_att) * n_items,
                ) for channels, factor, n_items, n_att in zip(*sequences)
            ],
            skip_t=skip_t,
            attention_features=attention_features,
            attention_heads=attention_heads,
            # embedding_features=embedding_features,
            modulation_features=modulation_features,
            resnet_groups=resnet_groups
        )

    def forward(self, x: Tensor, time: Tensor):
        x = einops.rearrange(x, 'b (x t) -> b t x', t=self.token_size)
        x = self._net(x, time=time)
        x = einops.rearrange(x, 'b t x -> b (x t)', t=self.token_size)
        return x

# unet = UNet(
#     dim=2,
#     in_channels=2,
#     channels=[128, 256, 512, 1024],
#     factors=[2, 2, 2, 2],
#     items=[2, 2, 2, 2],
#     attentions=[0, 0, 0, 1],
#     cross_attentions=[1, 1, 1, 1],
#     attention_features=64,
#     attention_heads=8,
#     embedding_features=768,
#     use_classifier_free_guidance=False
# )
# x = torch.randn(2, 2, 64, 64)
# time = [0.2, 0.5]
# embedding = torch.randn(2, 512, 768)
# y = unet(x, time=time, embedding=embedding) # [2, 2, 64, 64]