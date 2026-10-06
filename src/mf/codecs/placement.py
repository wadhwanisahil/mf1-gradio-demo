"""Move frozen codec inputs and outputs across devices without moving MF."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DeviceCodec(nn.Module):
    """Keep a frozen codec on its own device and return results to the caller."""

    def __init__(self, codec: nn.Module, device: torch.device) -> None:
        super().__init__()
        self.codec = codec
        self.device = device

    def encode(self, *inputs: Tensor) -> Tensor:
        output = self.codec.encode(*(value.to(self.device) for value in inputs))
        return output.to(inputs[0].device)

    def decode(self, inputs: Tensor) -> Tensor:
        output = self.codec.decode(inputs.to(self.device))
        return output.to(inputs.device)
