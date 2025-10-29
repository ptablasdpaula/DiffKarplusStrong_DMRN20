from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlpc import sample_wise_lpc

LAGRANGE_ORDER = 5
_EPS = 1e-12

class DiffKSBase(nn.Module):
    """
    Base class for differentiable Karplus–Strong with time-varying Lagrange fractional delay
    and first-order loop filter (FIR and IIR variants implemented in subclasses).

    Shapes:
      f0:     [B, N]   fractional period in *samples*
      input:  [B, N]   mono waveform
      l_b:    [B, N, 2] "logits" or controls to be mapped to taps by design_loop()
                        (interpreted here as [g, p] in [0,1], then mapped → [b0, a1])

      Where B is batch_size, N samples length of mono waveform.
      Where g stands for decay and p is the ratio between b0 and a1.

    Defaults:
      device: torch.device("cpu")
      dtype:  torch.float32
      Lagrange order: fixed to 5
    """

    def __init__(
        self,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @staticmethod
    def constrain_logits(l_b: torch.Tensor) -> torch.Tensor:
        """Optional hook to transform raw logits into constrained values.
        Default is a no-op; subclasses may override.
        """
        return l_b

    def forward(
        self,
        f0: torch.Tensor,         # [B, N] in samples (fractional period)
        input: torch.Tensor,      # [B, N] mono waveform
        l_b: torch.Tensor,        # [B, N, 2] controls → mapped to taps [b0, a1]
    ) -> torch.Tensor:
        self._basic_shape_checks(f0, input, l_b)

        taps = self.design_loop(l_b).to(self.dtype)
        f0   = f0.to(self.dtype)
        x    = input.to(self.dtype)

        # Apply loop-dependent f0 phase correction
        f0_corr = self.tune_f0(f0=f0, taps=taps)

        # Build coefficient matrix A and (possibly) pre-filtered excitation x
        A, x_eff = self.compute_resonator_matrix(f0=f0_corr, taps=taps, x=x)

        # Run sample-wise LPC
        y = sample_wise_lpc(x_eff, A)
        return y.to(self.dtype)

    def design_loop(self, l_b: torch.Tensor) -> torch.Tensor:
        """
        Map user controls l_b[..., (g, p)] ∈ [0,1]^2 to loop taps [b0, a1].
        """
        raise NotImplementedError

    def compute_resonator_matrix(
        self,
        *,
        f0: torch.Tensor,   # [B, N] corrected fractional delay (samples)
        taps: torch.Tensor, # [B, N, 2] = [b0, a1]
        x: torch.Tensor,    # [B, N] excitation (may be pre-filtered for IIR)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (A, x_eff), where:
          A: [B, N, K] coefficient matrix for the sample-wise LPC kernel
          x_eff: [B, N] possibly pre-filtered excitation (IIR case)
        """
        raise NotImplementedError

    def tune_f0(self, *, f0: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        """Subclass must implement loop-filter choice-dependant phase correction."""
        raise NotImplementedError

    def _lagrange_weights(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute (vectorised) Lagrange weights of length L+1 for per-sample fractional offset alpha.
        alpha: [B, N] with 0 ≤ alpha < 1 + L//2 (since we re-centre around z_center)
        returns: [B, N, L+1]
        """
        # j = 0..L
        j = torch.arange(LAGRANGE_ORDER + 1, device=self.device, dtype=self.dtype)  # [L+1]
        # For each k, w_k = Π_{m≠k} (alpha - m) / (k - m)
        # We do it by broadcasting (alpha[..., None] - j[None, None, :]) and masking the diagonal.
        u = alpha.unsqueeze(-1) - j  # [B, N, L+1]
        # Build denominators (k - m)
        k = j.view(LAGRANGE_ORDER + 1, 1)            # [L+1, 1]
        m = j.view(1, LAGRANGE_ORDER + 1)            # [1, L+1]
        denom = (k - m)                      # [L+1, L+1]
        mask = ~torch.eye(LAGRANGE_ORDER + 1, dtype=torch.bool, device=self.device)  # [L+1, L+1]

        # Numerator product excluding diagonal
        num = torch.where(mask, u.unsqueeze(-2), torch.ones_like(u.unsqueeze(-2)))          # [B, N, L+1, L+1]
        den = torch.where(mask, denom.to(self.dtype), torch.ones_like(denom).to(self.dtype))# [L+1, L+1]

        weights = (num / den).prod(dim=-1)  # [B, N, L+1]
        return weights

    @staticmethod
    def _basic_shape_checks(f0: torch.Tensor, x: torch.Tensor, l_b: torch.Tensor) -> None:
        assert f0.dim() == 2 and x.dim() == 2, "f0 and input must be [B, N]"
        assert l_b.dim() == 3 and l_b.size(-1) == 2, "l_b must be [B, N, 2]"
        B, N = x.shape
        assert f0.shape == (B, N), "f0 must match input shape"
        assert l_b.shape[:2] == (B, N), "l_b must match input shape"

    def _allocate_A(self, f0: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Allocate A with enough room to place Lagrange block plus a couple of guard taps.
        """
        B, N = f0.shape
        max_z = torch.floor(f0).to(torch.long).max()
        K = int(max_z.item()) + LAGRANGE_ORDER + 3
        a_mat = torch.zeros((B, N, K), device=self.device, dtype=self.dtype)
        return a_mat, K

    def _lagrange_center_and_weights(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.floor(f0).to(torch.long)
        z_center = z - (LAGRANGE_ORDER // 2)
        alpha = (f0 - z_center.to(self.dtype))
        w = self._lagrange_weights(alpha)
        return z_center, w

    @staticmethod
    def _scatter_block_(a_mat: torch.Tensor, z_center: torch.Tensor, block: torch.Tensor, k_cols: int) -> None:
        block_len = block.size(-1)
        device = a_mat.device
        idxs = z_center.unsqueeze(-1) + torch.arange(block_len, device=device)
        a_mat.scatter_add_(dim=2, index=idxs.clamp_max(k_cols - 1), src=block)

class DiffKSIIRLoop(DiffKSBase):
    """Order-1 IIR loop inside the KS feedback path."""

    def design_loop(self, l_b: torch.Tensor) -> torch.Tensor:
        l_b = self.constrain_logits(l_b)
        g = l_b[..., 0].to(self.dtype)
        p = l_b[..., 1].to(self.dtype)
        assert p.max().item() <= 1.0
        assert g.max().item() <= 1.0
        a1 = p
        b0 = (1.0 - a1) * g
        taps = torch.stack([b0, a1], dim=-1)  # [B, N, 2]
        assert torch.all(taps.abs() <= 1.0)
        return taps

    def tune_f0(self, *, f0: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        a1 = taps[..., 1]
        omega = 2 * torch.pi / f0
        cosw = torch.cos(omega)
        sinw = torch.sin(omega)
        real1 = 1.0 - a1 * cosw
        imag1 = a1 * sinw
        phase = torch.atan2(imag1, real1)
        p_a = phase / (omega + _EPS)
        c = 1.0
        return f0 - (c + p_a)

    def compute_resonator_matrix(
        self,
        *,
        f0: torch.Tensor,
        taps: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_samples = f0.shape
        b0 = taps[..., 0]
        a1 = taps[..., 1]

        # Premultiply excitation for AR(1) denominator (time-varying)
        x_eff = x.clone()
        x_eff[:, 0] = x[:, 0] * (1.0 - a1[:, 0])
        if n_samples > 1:
            x_eff[:, 1:] = x[:, 1:] - a1[:, 1:] * x[:, :-1]

        # Build A
        a_mat, k_cols = self._allocate_A(f0)

        # Indices for Lagrange kernel centered window
        z_center, w = self._lagrange_center_and_weights(f0)

        # IIR branch: block = -(b0 * w) padded with one trailing 0  → length L+2
        block = -(b0.unsqueeze(-1) * w)                    # [B, N, L+1]
        block = F.pad(block, (0, 1))                       # [B, N, L+2]

        # Scatter-add into A at indices z_center + [0..L+1]
        self._scatter_block_(a_mat, z_center, block, k_cols)

        # AR(1) term at index 1
        a_mat[:, :, 1] += -a1

        return a_mat, x_eff

class DiffKSFIRLoop(DiffKSBase):
    """Order-1 FIR loop inside the KS feedback path."""

    def design_loop(self, l_b: torch.Tensor) -> torch.Tensor:
        l_b = self.constrain_logits(l_b)
        g = l_b[..., 0].to(self.dtype)
        p = l_b[..., 1].to(self.dtype)
        assert p.max().item() <= 1.0
        assert g.max().item() <= 1.0
        b0 = (1.0 - p) * g
        a1 = p * g
        taps = torch.stack([b0, a1], dim=-1)  # [B, N, 2]
        assert torch.all(taps.abs() <= 1.0)
        return taps

    def tune_f0(self, *, f0: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
        b0 = taps[..., 0]
        a1 = taps[..., 1]
        omega = 2 * torch.pi / f0
        cosw = torch.cos(omega)
        sinw = torch.sin(omega)
        realn = b0 + a1 * cosw
        imagn = -a1 * sinw
        phase = torch.atan2(imagn, realn)
        p_a = -phase / (omega + _EPS)
        c = 1.0
        return f0 - (c + p_a)

    def compute_resonator_matrix(
        self,
        *,
        f0: torch.Tensor,
        taps: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _batch_size, _n_samples = f0.shape
        b0 = taps[..., 0]
        a1 = taps[..., 1]

        a_mat, k_cols = self._allocate_A(f0)

        z_center, w = self._lagrange_center_and_weights(f0)

        # FIR branch: b0*w at center + a1*shifted(+1)
        core = (b0.unsqueeze(-1) * w)                      # [B, N, L+1]
        block = F.pad(-core, (0, 1)) + (a1.unsqueeze(-1) * F.pad(-core, (1, 0)))  # [B, N, L+2]

        self._scatter_block_(a_mat, z_center, block, k_cols)

        return a_mat, x

class LearnableCoefficientsMixin:
    """
    Mixin that maps unconstrained logits l_b to constrained controls:
      g = g_min + (1 - g_min) * sigmoid(g_raw)  in [g_min, 1.0]
      p = sigmoid(p_raw)                        in [0, 1]
    Override order does not matter as long as this mixin appears before the loop class
    in the MRO, so that its `constrain_logits` overrides the base no-op.
    """
    def __init__(self, *, g_min: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.g_min = float(g_min)
        self.g_span = 1.0 - self.g_min

    def constrain_logits(self, l_b: torch.Tensor) -> torch.Tensor:
        g_raw = l_b[..., 0]
        p_raw = l_b[..., 1]
        g = self.g_min + self.g_span * torch.sigmoid(g_raw)  # g ∈ [g_min, 1]
        p = torch.sigmoid(p_raw)                              # p ∈ [0, 1]
        return torch.stack([g, p], dim=-1)

class DiffKSIIRLoopLearnableCoefficients(LearnableCoefficientsMixin, DiffKSIIRLoop):
    """
    IIR loop with learnable (unconstrained) controls l_b.
    Uses LearnableCoefficientsMixin to constrain g and p via sigmoid.
    Pass `g_min` to constructor to change the lower bound (default 0.9).
    """
    pass

class DiffKSFIRLoopLearnableCoefficients(LearnableCoefficientsMixin, DiffKSFIRLoop):
    """
    FIR loop with learnable (unconstrained) controls l_b.
    Uses LearnableCoefficientsMixin to constrain g and p via sigmoid.
    Pass `g_min` to constructor to change the lower bound (default 0.9).
    """
    pass