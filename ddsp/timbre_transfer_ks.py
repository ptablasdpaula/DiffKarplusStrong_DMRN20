import torch
from torch import nn

from .diff_ks import DiffKSIIRLoopLearnableCoefficients
from .excitation_shaper import ExcitationShaper

class TimbreTransferKS(nn.Module):
    def __init__(
            self,
            *,
            device: torch.device = torch.device('cpu'),
            dtype: torch.dtype = torch.float32,
            sr: int = 16000,
        ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.sr = sr
        self.karplus_strong = DiffKSIIRLoopLearnableCoefficients(device=device, dtype=dtype)
        self.exc_shaper = ExcitationShaper(device=device, dtype=dtype, sr=sr)

    def forward(
            self,
            params: torch.Tensor, # [B, T, 6] Parameters to optimise (2 (diffks) + 4 (exc_shaper))
            f0: torch.Tensor, # [B, T]
            onsets: torch.Tensor,
            bursts: torch.Tensor,
            return_stats = False,
        ):
        params = params.to(device=self.device, dtype=self.dtype)

        if return_stats:
            shaped_exc, stats_exc = self.exc_shaper(params=params[..., :4], f0=f0, onsets=onsets, input=bursts, return_stats=True)
            y, stats_diffks = self.karplus_strong(f0=(self.sr/f0), input=shaped_exc, params=params[..., 4:], return_stats=True)
            return y, {"stats_exc": stats_exc, "stats_diffks": stats_diffks, "shaped_exc": shaped_exc}

        shaped_exc = self.exc_shaper(f0=f0, input=bursts, params=params[..., :4], onsets=onsets,)
        y = self.karplus_strong(f0=(self.sr/f0), input=shaped_exc, params=params[..., 4:])
        return y