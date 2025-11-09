import torch
import torch.nn as nn

import sys, os
from .utils import map_logspace, segment_ids_from_onsets, piecewise_average_by_segments

tb303_util_path = os.path.join(
    os.path.dirname(__file__), "..", "TB303", "acid_ddsp"
)
tb303_util_path = os.path.abspath(tb303_util_path)

if tb303_util_path not in sys.path:
    sys.path.insert(0, tb303_util_path)

from TB303.acid_ddsp.filters import TimeVaryingLPBiquad

def tv_all_zero_fir(x: torch.Tensor,    # [B, T] mono input
                    A: torch.Tensor     # [B, T, D] where D is filter order
                    ) -> torch.Tensor:  # [B, T]
    B, T = x.shape
    D = A.shape[2]

    x_padded = torch.cat([torch.zeros(B, D, device=x.device, dtype=x.dtype), x], dim=1)  # [B, D+T]
    shifts = torch.stack([x_padded[:, D - k:D - k + T] for k in range(1, D + 1)], dim=2)  # [B, T, D]
    return x + torch.sum(A * shifts, dim=2) # Direct term plus weighted sum of delayed taps

def pluck_comb(f0,      # [B, T] length (L) of string
               mu,      # [B, T] ratio of L (0.5 -> middle of string)
               input,   # [B, T]
               ) -> torch.Tensor: # [B, T]
    B, T = input.shape
    batch_indices = torch.arange(B, device=input.device).view(-1, 1).expand(-1, T)
    sample_indices = torch.arange(T, device=input.device).view(1, -1).expand(B, -1)

    pluck_position = f0 * mu
    coeff_vector_size = int(torch.ceil(pluck_position.max()).item()) + 2

    z_l = torch.floor(pluck_position).long()
    alfa = pluck_position - z_l

    A = torch.zeros((B, T, coeff_vector_size), device=input.device, dtype=input.dtype)
    A[batch_indices, sample_indices, z_l] = - (1 - alfa)
    A[batch_indices, sample_indices, z_l + 1] = -alfa

    return tv_all_zero_fir(input, A)

class ExcitationShaper(nn.Module):
    """
    Unified excitation shaper.

    Responsibilities (toggled at init):
      - average_per_onset: if True, average params per segment (requires segment_ids)
      - mod_parameters: if True, map params -> physical params via internal mapping
      - always performs DSP

    Forward signature always expects params as [B, T, 4].
    Stats are returned only when `mod_parameters=True` and `return_stats=True`.
    """
    def __init__(
        self,
        *,
        sr: int = 16000,
        max_q: float = 2.0,
        min_d: float = 0.1,
        max_d: float = 2.0,
        mod_parameters: bool = True,
        average_per_onset: bool = True,
        device : torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mod_parameters = mod_parameters
        self.average_per_onset = average_per_onset
        self.min_d = min_d
        self.max_d = max_d

        self.dynamics_filter = TimeVaryingLPBiquad(
            min_w=2.0 * torch.pi * 20.0 / sr,
            max_q=max_q,
        ).to(device=device, dtype=dtype)

    def _average_params(self, params, onsets):
        if not self.average_per_onset:
            return params
        assert onsets is not None, "onsets must be provided when average_per_onset=True"
        B, T, _ = params.shape
        segment_ids = segment_ids_from_onsets(onsets, T, device=params.device)
        return piecewise_average_by_segments(params, segment_ids)

    def _constrain_params(self, logits: torch.Tensor):
        if not self.mod_parameters:
            return logits.unbind(dim=-1)
        distance = map_logspace(torch.sigmoid(logits[..., 0]), min=self.min_d, max=self.max_d,)
        w_mod = torch.sigmoid(logits[..., 1])
        q_mod = torch.sigmoid(logits[..., 2])
        mu = torch.sigmoid(logits[..., 3])
        return distance, w_mod, q_mod, mu

    def _run_dsp(self, f0, input, distance, w_mod, q_mod, mu):
        x = input * distance
        x = pluck_comb(f0, mu, x)
        y, *_ = self.dynamics_filter(x=x, w_mod_sig=w_mod, q_mod_sig=q_mod)
        return y

    def _collect_output(self, y, distance, w_mod, q_mod, mu, return_stats):
        if not (return_stats and self.mod_parameters):
            return y
        stats = {
            "distance": distance.detach(),
            "w_mod": w_mod.detach(),
            "q_mod": q_mod.detach(),
            "mu": mu.detach(),
        }
        return y, stats

    def forward(
        self,
        *,
        f0: torch.Tensor,             # [B, T]
        input: torch.Tensor,          # [B, T]
        params: torch.Tensor,         # [B, T, 4]
        onsets: torch.Tensor,
        return_stats: bool = False,
    ):
        input = input.to(device=self.device, dtype=self.dtype)
        f0 = f0.to(device=self.device, dtype=self.dtype)
        params = params.to(device=self.device, dtype=self.dtype)
        onsets = onsets.to(device=self.device)

        assert input.dim() == 2, "input must be [B, T]"
        B, T = input.shape
        assert f0.shape == (B, T)
        assert params.dim() == 3 and params.shape[:2] == (B, T) and params.size(-1) == 4

        params = self._average_params(params, onsets)
        distance, w_mod, q_mod, mu = self._constrain_params(params)
        y = self._run_dsp(f0, input, distance, w_mod, q_mod, mu)
        return self._collect_output(y, distance, w_mod, q_mod, mu, return_stats)