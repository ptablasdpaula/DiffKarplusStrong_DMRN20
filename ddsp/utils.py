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

    def forward(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """
        x, x_target: [B, 1, T]
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

            # p-norm across time/frequency
            dist = torch.linalg.vector_norm(log_Sx_target - log_Sx, ord=self.p, dim=(-2, -1))
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