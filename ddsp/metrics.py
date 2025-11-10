import torch
import torch.nn as nn
import scipy.signal
from typing import List, Optional
from torchaudio.transforms import MFCC

class MFCCDistance(nn.Module):
    def __init__(
        self,
        sr: int,
        log_mels: bool = True,
        n_fft: int = 2048,
        hop_len: int = 512,
        n_mels: int = 128,
        n_mfcc: int = 40,
        p: int = 1,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.p = p

        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            log_mels=log_mels,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_len,
                "n_mels": n_mels,
            },
        )
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape == x_target.shape
        if self.p == 1:
            return self.l1(self.mfcc(x), self.mfcc(x_target))
        elif self.p == 2:
            return self.mse(self.mfcc(x), self.mfcc(x_target))
        else:
            raise ValueError(f"Unknown p value: {self.p}")

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