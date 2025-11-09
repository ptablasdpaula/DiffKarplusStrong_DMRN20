import librosa
import numpy as np
import torch
import torch.nn as nn
import scipy.signal
from typing import List, Optional

def detect_f0(x, sr, frame_length=4096, hop_length=4096):
    return librosa.yin(y=x,fmin=70.0,fmax=500.0,sr=sr,frame_length=frame_length,hop_length=hop_length,center=False,trough_threshold=0.01)

def detect_onsets(x, sr, hop_length=512, pad_ms=50):
    pad = int(round((pad_ms / 1000.0) * sr))
    x_pad = np.pad(x, (pad, 0), mode="constant")
    onset_frames = librosa.onset.onset_detect(y=x_pad,sr=sr,hop_length=hop_length,backtrack=True,units="frames")
    onset_frames = np.unique(np.asarray(onset_frames, dtype=int))
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    onset_samples = onset_samples - pad
    onset_samples = onset_samples[(onset_samples >= 0) & (onset_samples < len(x))]
    return onset_samples.astype(int)

def bursts_at_onsets(x, sr, onset_samples, f0, burst_range=1.0, frame_hop=4096, seed=42):
    n_samples = len(x)
    noise_signal = np.zeros_like(x, dtype=np.float32)
    random_generator = np.random.default_rng(seed)
    for onset in onset_samples:
        frame_idx = min(int(onset // frame_hop), len(f0) - 1)
        f0_val = f0[frame_idx]
        burst_len = int(np.round(sr / f0_val))
        end = min(onset + burst_len, n_samples)
        burst = random_generator.random(end - onset).astype(np.float32)
        burst -= burst.mean()
        burst /= np.max(np.abs(burst))
        burst *= burst_range
        noise_signal[onset:end] += burst
    return noise_signal

def segment_ids_from_onsets(onsets: torch.Tensor, T: int, device=None) -> torch.Tensor:
    """Build [B, T] segment-id tensor from batched onset indices.
    Each row in `onsets` corresponds to one batch of onset sample indices.
    Labels increase by 1 at each onset; the first segment starts at 0.
    """
    assert onsets.dim() in (1, 2), "onsets must be [N_onsets] or [B, N_onsets]"
    if onsets.dim() == 1:
        onsets = onsets.unsqueeze(0)
    B = onsets.shape[0]
    device = device if device is not None else onsets.device
    seg_all = []
    for b in range(B):
        onsets_b = onsets[b].to(torch.long)
        if onsets_b.numel() == 0:
            seg_all.append(torch.zeros(T, dtype=torch.long, device=device))
            continue
        onsets_b = torch.unique(torch.clamp(onsets_b, 0, T))
        if onsets_b[0].item() != 0:
            onsets_b = torch.cat([torch.tensor([0], device=device, dtype=torch.long), onsets_b], dim=0)
        change = torch.zeros(T, dtype=torch.long, device=device)
        change.index_fill_(0, onsets_b, 1)
        seg_ids = torch.cumsum(change, dim=0) - 1
        seg_all.append(seg_ids)
    return torch.stack(seg_all, dim=0)

def piecewise_average_by_segments(values: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """
    Average `values` within segments defined by `segment_ids` and return a piece-wise
    constant tensor with the same shape.

    Args:
        values: Tensor [B, T, C]
        segment_ids: LongTensor [B, T]
    Returns:
        Tensor [B, T, C] with per-segment averages broadcast back to samples.
    """
    assert values.dim() == 3, "values must be [B, T, C]"
    assert segment_ids.dim() == 2, "segment_ids must be [B, T]"
    B, T, C = values.shape
    assert segment_ids.shape == (B, T), "segment_ids must match [B, T]"

    out = torch.empty_like(values)
    for b in range(B):
        x_bt = values[b]                 # [T, C]
        labels_bt = segment_ids[b].to(torch.long)  # [T]
        device, dtype = x_bt.device, x_bt.dtype
        if T == 0:
            out[b] = x_bt
            continue
        # Change-points → run IDs in [0, S-1]
        newseg = torch.ones(T, dtype=torch.bool, device=device)
        newseg[1:] = labels_bt[1:] != labels_bt[:-1]
        run_ids = newseg.to(torch.long).cumsum(dim=0) - 1  # [T]
        S = int(run_ids.max().item()) + 1
        # Scatter-add sums and counts per run
        idx = run_ids.unsqueeze(1).expand(-1, C)  # [T, C]
        sums = torch.zeros(S, C, device=device, dtype=dtype)
        sums.scatter_add_(0, idx, x_bt)
        counts = torch.zeros(S, 1, device=device, dtype=dtype)
        counts.scatter_add_(0, run_ids.unsqueeze(1), torch.ones(T, 1, device=device, dtype=dtype))
        means = sums / counts.clamp_min(1)
        # Map means back to per-sample
        out[b] = means[run_ids]
    return out

def map_logspace(x: torch.Tensor, min: float, max: float):
    return torch.exp((torch.log(torch.tensor(max)) - torch.log(torch.tensor(min))) * x + torch.log(torch.tensor(min)))

class LogMSSLoss(nn.Module):
    """
    Multi-Scale Log-Magnitude Spectral Loss (Taken from https://github.com/christhetree/scrapl-ddsp/blob/main/experiments/losses.py)
    Defaults to the hyperparameters from "Multi-Scale Spectral Loss Revisited" by Schwär. S. and Müller M.
    """

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = "flat_top",
        log_mag_eps: float = 1.0,
        gamma: float = 1.0,
        p: int = 2,
    ):
        super().__init__()

        # Default parameters (smooth multi-scale configuration)
        if win_lengths is None:
            win_lengths = [67, 127, 257, 509, 1021, 2053]
        if fft_sizes is None:
            fft_sizes = [67, 127, 257, 509, 1021, 2053]
        if hop_sizes is None:
            hop_sizes = [33, 63, 128, 254, 510, 1026]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        self.log_mag_eps = log_mag_eps
        self.gamma = gamma
        self.p = p

        # Precompute and register window buffers
        for win_length in win_lengths:
            w = self.make_window(window, win_length)
            self.register_buffer(f"window_{win_length}", w)

    def interp_to_sample_rate(f0_frames: np.ndarray, T: int, sr: int, hop: int) -> np.ndarray:
        """Interpolate frame-rate f0 (Hz) to sample rate (length T)."""
        t_audio = np.arange(T) / sr
        t_f0 = (np.arange(len(f0_frames)) * hop) / sr
        return np.interp(t_audio, t_f0, f0_frames)

    def forward(self, x: torch.Tensor, x_target: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        x, x_target: [B, 1, T]
        normalize: if True, normalize the Lp distance by the number of time-frequency bins per STFT scale.

        Returns scalar loss.
        """
        assert x.ndim == x_target.ndim == 3, "inputs must be [B, 1, T]"
        assert x.size(1) == x_target.size(1) == 1

        x = x.squeeze(1)
        x_target = x_target.squeeze(1)
        dists = []

        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            win = getattr(self, f"window_{win_length}")

            Sx = torch.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()
            Sx_target = torch.stft(
                x_target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()

            # Log-magnitude transform
            if self.log_mag_eps == 1.0:
                log_Sx = torch.log1p(self.gamma * Sx)
                log_Sx_target = torch.log1p(self.gamma * Sx_target)
            else:
                log_Sx = torch.log(self.gamma * Sx + self.log_mag_eps)
                log_Sx_target = torch.log(self.gamma * Sx_target + self.log_mag_eps)

            diff = log_Sx_target - log_Sx

            if not normalize:
                dist = torch.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1))
            else:
                n_time, n_freq = diff.shape[-2], diff.shape[-1]
                if self.p == 1:
                    dist = diff.abs().mean(dim=(-2, -1))
                else:
                    dist = torch.linalg.vector_norm(diff, ord=self.p, dim=(-2, -1)) / (n_time * n_freq) ** (1 / self.p)

            dists.append(dist)

        return torch.stack(dists, dim=1).sum(dim=1).mean()

    @staticmethod
    def make_window(window: str, n: int) -> torch.Tensor:
        """Create and return a window tensor."""
        if window == "rect":
            return torch.ones(n)
        elif window == "hann":
            return torch.hann_window(n)
        elif window == "flat_top":
            w = scipy.signal.windows.flattop(n, sym=False)
            return torch.from_numpy(w).float()
        else:
            raise ValueError(f"Unknown window type: {window}")