import torch
import torch.nn as nn
from TB303.acid_ddsp.filters import TimeVaryingLPBiquad

class DynamicsShaper(nn.Module):
    """
    Shape burst excitation with piecewise-constant dynamics & a time-varying biquad.

    Forward inputs
    --------------
    noise_bursts : Tensor [B, T]
        Excitation (e.g., bursts_at_onsets output converted to torch).
    segment_ids : LongTensor [B, T]
        Segment labels per sample. Consecutive samples with the same label
        belong to the same segment (labels need not be 0..K-1 globally; they
        are re-run-labeled internally per batch item).
    logits : Tensor [B, T, 3]
        Unbounded controls per sample. Channels are:
          [:, :, 0] -> gain logit (mapped to [gain_min, gain_max] via sigmoid)
          [:, :, 1] -> w_mod logit for TimeVaryingLPBiquad (sigmoid -> [0,1])
          [:, :, 2] -> q_mod logit for TimeVaryingLPBiquad (sigmoid -> [0,1])
    return_stats : bool (default False)
        If True, also return a stats dict with per-sample controls.

    Behavior
    --------
    Average logits within labeled segments so they are static per segment.
    Apply per-sample gain to the excitation.
    Run TimeVaryingLPBiquad with w/q modulators (per-sample, no interpolation).

    Returns
    -------
    y : Tensor [B, T]
        Filtered, gain-shaped excitation.
    stats : dict (optional)
        Only returned when return_stats=True. Contains:
          - "gain": Tensor [B, T]
          - "w_mod": Tensor [B, T]
          - "q_mod": Tensor [B, T]
          - "segment_ids": LongTensor [B, T]
    """
    def __init__(
        self,
        gain_min: float = 0.1,
        gain_max: float = 2.0,
        min_q: float = 0.0707,
        max_q: float = 2.0,
        sr: int = 16000,
    ):
        super().__init__()
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

        # Time-varying LP biquad (expects modulators in [0,1])
        self.lpf = TimeVaryingLPBiquad(
            min_w=2.0 * torch.pi * 20.0 / sr,  # ~20 Hz (rad/sample)
            max_w=float(torch.pi),             # Nyquist (rad/sample)
            min_q=min_q,
            max_q=max_q,
            modulate_log_w=True,
            modulate_log_q=True,
        )

    @staticmethod
    def _avg_by_run_ids(x_bt: torch.Tensor, labels_bt: torch.Tensor) -> torch.Tensor:
        """
        Average x within *runs* defined by labels_bt for a single batch item.
        The labels are converted into run IDs (change points) so they do not
        need to be contiguous or start at zero.

        Args:
            x_bt:      [T, C]
            labels_bt: [T] (int64)
        Returns:
            out_bt:    [T, C] piecewise-constant averages per run
        """
        T, C = x_bt.shape
        if T == 0:
            return x_bt
        labels_bt = labels_bt.to(torch.long)
        device = x_bt.device
        dtype = x_bt.dtype

        # Identify change points → run IDs in [0, S-1]
        newseg = torch.ones(T, dtype=torch.bool, device=device)
        newseg[1:] = labels_bt[1:] != labels_bt[:-1]
        run_ids = newseg.to(torch.long).cumsum(dim=0) - 1  # [T]
        S = int(run_ids.max().item()) + 1

        # Scatter-add sums and counts per run (vectorized)
        idx = run_ids.unsqueeze(1).expand(-1, C)  # [T, C]
        sums = torch.zeros(S, C, device=device, dtype=dtype)
        sums.scatter_add_(0, idx, x_bt)

        counts = torch.zeros(S, 1, device=device, dtype=dtype)
        counts.scatter_add_(0, run_ids.unsqueeze(1), torch.ones(T, 1, device=device, dtype=dtype))
        means = sums / counts.clamp_min(1)

        # Map back to per-sample
        out = means[run_ids]
        return out

    def _make_piecewise_constant(self, logits_raw: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Return logits averaged within labeled segments: [B, T, 3]
        """
        assert logits_raw.dim() == 3 and logits_raw.size(-1) == 3
        assert segment_ids.dim() == 2
        B, T, C = logits_raw.shape
        assert segment_ids.shape == (B, T)
        pcs = []
        for b in range(B):
            pcs.append(self._avg_by_run_ids(logits_raw[b], segment_ids[b]).unsqueeze(0))
        return torch.cat(pcs, dim=0)

    def forward(
        self,
        noise_bursts: torch.Tensor,   # [B, T]
        segment_ids: torch.Tensor,    # [B, T] (long)
        logits: torch.Tensor,         # [B, T, 3]
        return_stats: bool = False,   # when True, also return a stats dict
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        assert noise_bursts.dim() == 2, "noise_bursts must be [B, T]"
        assert logits.dim() == 3 and logits.size(-1) == 3, "logits must be [B, T, 3]"
        B, T = noise_bursts.shape
        assert logits.shape[:2] == (B, T), "logits and noise_bursts must share [B, T]"
        assert segment_ids.shape == (B, T), "segment_ids must be [B, T]"
        assert segment_ids.dtype in (torch.int64, torch.long, torch.int32), "segment_ids must be integer"
        segment_ids = segment_ids.to(torch.long)

        logits_seg = self._make_piecewise_constant(logits, segment_ids)  # [B, T, 3]

        gain = torch.sigmoid(logits_seg[..., 0])                         # [B, T]
        gain = self.gain_min + (self.gain_max - self.gain_min) * gain
        w_mod = torch.sigmoid(logits_seg[..., 1])                        # [B, T] in [0,1]
        q_mod = torch.sigmoid(logits_seg[..., 2])                        # [B, T] in [0,1]

        x = noise_bursts * gain
        y, *_ = self.lpf(
            x=x,                          # [B, T]
            w_mod_sig=w_mod,              # [B, T]
            q_mod_sig=q_mod,              # [B, T]
            interp_coeff=False
        )

        if not return_stats:
            return y

        stats = {
            "gain": gain.detach(),
            "w_mod": w_mod.detach(),
            "q_mod": q_mod.detach(),
            "segment_ids": segment_ids.detach(),
        }
        return y, stats

def segment_ids_from_onsets(onsets: torch.Tensor, T: int, device=None) -> torch.Tensor:
    """Utility: build a [T] segment-id vector from onset sample indices.
    Labels increase by 1 at each onset; the first segment starts at 0.
    `onsets` may be 1-D LongTensor (assumed sorted); 0 will be added if missing.
    """
    device = device if device is not None else onsets.device
    onsets = onsets.to(torch.long)
    if onsets.numel() == 0:
        return torch.zeros(T, dtype=torch.long, device=device)
    onsets = torch.unique(torch.clamp(onsets, 0, T))
    if onsets[0].item() != 0:
        onsets = torch.cat([torch.tensor([0], device=device, dtype=torch.long), onsets], dim=0)
    # Binary change mask
    change = torch.zeros(T, dtype=torch.long, device=device)
    change.index_fill_(0, onsets, 1)
    # Cumulative sum gives segment ids
    seg_ids = torch.cumsum(change, dim=0) - 1
    return seg_ids