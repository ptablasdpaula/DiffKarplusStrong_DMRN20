import argparse
from dirs import (
    EXPERIMENTS_AUDIO_GRADIENT,
    EXPERIMENTS_AUDIO_TARGET,
    EXPERIMENTS_RESULTS_TABLES,
)
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import pandas as pd
import torchaudio
import os
import random
import numpy as np

from ddsp.timbre_transfer_ks import TimbreTransferKS
from ddsp.utils import LogMSSLoss
from data.preprocess_subset import GuitarAcousticDataset

# -----------------------------
# Constants & simple helpers
# -----------------------------
SR: int = 16000
SPLIT: str = "test"
ROOT: str = "nsynth_preprocessed"

# MIDI boundaries: E2=40, E3=52, E4=64, E5=76, E6=88
OCT_BOUNDS: Dict[int, Tuple[int, int]] = {
    1: (40, 52),   # [40..51]
    2: (52, 64),   # [52..63]
    3: (64, 76),   # [64..75]
    4: (76, 89),   # [76..88], upper bound exclusive
}

# Discrete NSynth velocities & labels
VEL_LABELS: List[str] = ["p", "mp", "mf", "f", "ff"]
VEL_VALUES: List[int] = [25, 50, 75, 100, 127]


def make_deterministic(seed: int = 1337) -> None:
    """Set seeds and deterministic flags for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # cuBLAS determinism (only matters on CUDA)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Avoid TF32 variation
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # Error on non-deterministic ops (warn_only=True to avoid crashing on ops we can't control)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    """Prefer CUDA, else CPU."""
    if torch.cuda.is_available():
        print("Using CUDA device for computation.")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Optimise parameters for TimbreTransferKS")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size")
    p.add_argument("--num_frames", type=int, default=16, help="Frames per clip (F)")
    p.add_argument("--num_iters", type=int, default=150, help="Optimisation steps per batch")
    p.add_argument("--learning_rate", type=float, default=1e-1, help="Learning rate")
    p.add_argument("--num_batches", type=int, default=None,
                   help="Max batches to run per (octave, velocity) combo (for quick tests)")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    p.add_argument("--es_window", type=int, default=20,
                   help="Early stopping window (iterations) for mean/variance checks")
    p.add_argument("--es_mean_pct", type=float, default=5.0,
                   help="Stop if mean loss improves < this percent over the last window")
    p.add_argument("--es_var_pct", type=float, default=5.0,
                   help="Stop if variance change < this percent over the last window")
    p.add_argument("--min_iters", type=int, default=20,
                   help="Minimum iterations before early stopping is considered")
    return p.parse_args()


# -----------------------------
# Dataloader utilities
# -----------------------------
def collate_examples(batch):
    """Collate a list of (audio[T], f0_hz[T], onsets[K], bursts[T]) into padded tensors."""
    audios, f0s, onsets_list, bursts = zip(*batch)

    x_tgt = torch.stack(audios)   # [B, T]
    f0 = torch.stack(f0s)         # [B, T]
    bursts = torch.stack(bursts)  # [B, T]

    max_k = max(o.numel() for o in onsets_list)
    onsets = torch.full((len(onsets_list), max_k), -1, dtype=torch.long)
    for i, o in enumerate(onsets_list):
        onsets[i, :o.numel()] = o
    return x_tgt, f0, onsets, bursts


def indices_for_octave_velocity(ds: GuitarAcousticDataset, octave: int, velocity_label: str) -> List[int]:
    """Return dataset indices that match (octave, velocity)."""
    lo, hi = OCT_BOUNDS[octave]
    vel_val = VEL_VALUES[VEL_LABELS.index(velocity_label)]
    keep: List[int] = []
    for i, k in enumerate(ds.keys):
        meta = ds.meta[k]
        midi = int(meta.get("midi_pitch", -1))
        vel = int(meta.get("midi_velocity", -1))
        if (lo <= midi < hi) and (vel == vel_val):
            keep.append(i)
    return sorted(keep, key=lambda idx: ds.keys[idx])


# -----------------------------
# Per-sample loss for early stopping
# -----------------------------
def _batch_per_sample_loss(loss_fn: nn.Module, y_pred_bt: torch.Tensor, x_tgt_bt: torch.Tensor) -> torch.Tensor:
    """Compute per-sample non-normalized loss using the training criterion."""
    B = y_pred_bt.shape[0]
    vals = []
    with torch.no_grad():
        for b in range(B):
            yb = y_pred_bt[b:b+1].unsqueeze(1)
            xb = x_tgt_bt[b:b+1].unsqueeze(1)
            lb = loss_fn(yb, xb)
            if not isinstance(lb, torch.Tensor):
                lb = torch.tensor(lb, device=yb.device)
            vals.append(lb.detach().reshape(()))
    return torch.stack(vals).float()


# -----------------------------
# Optimization loop (per loader) with early stopping
# -----------------------------
def run_optim_over_loader(
    loader: DataLoader,
    model: TimbreTransferKS,
    loss_fn: LogMSSLoss,
    num_frames: int,
    num_iters: int,
    lr: float,
    device: torch.device,
    dtype: torch.dtype,
    octave: int,
    velocity: str,
    save_audio: bool = True,
    *,
    es_window: int = 10,
    es_mean_pct: float = 10.0,
    es_var_pct: float = 10.0,
    min_iters: int = 20,
) -> Tuple[List[List[float]], List[torch.Tensor], List[torch.Tensor]]:
    """Run optimization with dual early stopping based on mean and variance improvement."""
    all_batch_losses: List[List[float]] = []
    batch_final_preds: List[torch.Tensor] = []
    batch_final_targets: List[torch.Tensor] = []

    AUDIO_TARGET_DIR = EXPERIMENTS_AUDIO_TARGET
    AUDIO_GRADIENT_DIR = EXPERIMENTS_AUDIO_GRADIENT
    AUDIO_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_GRADIENT_DIR.mkdir(parents=True, exist_ok=True)

    for batch_id, (x_tgt, f0, onsets, bursts) in enumerate(loader, start=1):
        x_tgt = x_tgt.to(device=device, dtype=dtype)
        f0 = f0.to(device=device, dtype=dtype)
        onsets = onsets.to(device=device, dtype=torch.long)
        bursts = bursts.to(device=device, dtype=dtype)

        B, T = x_tgt.shape
        params_frames = nn.Parameter(torch.zeros(B, num_frames, 6, device=device, dtype=dtype))
        opt = optim.AdamW([params_frames], lr=lr)

        batch_losses: List[float] = []
        mean_hist: List[float] = []
        var_hist: List[float] = []
        early_stop_triggered = False

        pbar = tqdm(range(num_iters),
                    desc=f"Batch {batch_id} (octave={octave}, vel={velocity})",
                    leave=False, ncols=100)

        y_pred = None
        for iter_idx in pbar:
            opt.zero_grad()
            params_samples = F.interpolate(
                params_frames.permute(0, 2, 1),
                size=T,
                mode="linear",
                align_corners=True
            ).permute(0, 2, 1)

            y_pred = model(params=params_samples, f0=f0, onsets=onsets, bursts=bursts)
            loss = loss_fn(y_pred.unsqueeze(1), x_tgt.unsqueeze(1))
            loss.backward()
            opt.step()

            loss_scalar = float(loss.item())
            batch_losses.append(loss_scalar)

            ps_losses = _batch_per_sample_loss(loss_fn, y_pred, x_tgt)
            cur_mean = float(ps_losses.mean().item())
            cur_var = float(ps_losses.var(unbiased=False).item() if ps_losses.numel() > 1 else 0.0)
            mean_hist.append(cur_mean)
            var_hist.append(cur_var)
            pbar.set_postfix({"loss": f"{loss_scalar:.5f}", "μ": f"{cur_mean:.3f}", "σ²": f"{cur_var:.3f}"})

            have_windows = len(mean_hist) >= 2 * es_window
            if have_windows and (iter_idx + 1) >= min_iters:
                prev_mean = float(np.mean(mean_hist[-2*es_window:-es_window]))
                recent_mean = float(np.mean(mean_hist[-es_window:]))
                prev_var = float(np.mean(var_hist[-2*es_window:-es_window]))
                recent_var = float(np.mean(var_hist[-es_window:]))

                mean_improve_pct = 100.0 * (prev_mean - recent_mean) / max(abs(prev_mean), 1e-12)
                var_change_pct = 100.0 * abs(recent_var - prev_var) / max(abs(prev_var), 1e-12)

                if (mean_improve_pct < es_mean_pct) and (var_change_pct < es_var_pct):
                    early_stop_triggered = True

            if save_audio and (iter_idx == num_iters - 1 or early_stop_triggered):
                for b in range(y_pred.shape[0]):
                    torchaudio.save(
                        str(AUDIO_GRADIENT_DIR / f"oct{octave}_vel{velocity}_{b}.wav"),
                        y_pred[b].detach().cpu().unsqueeze(0),
                        SR
                    )
                    torchaudio.save(
                        str(AUDIO_TARGET_DIR / f"oct{octave}_vel{velocity}_{b}.wav"),
                        x_tgt[b].detach().cpu().unsqueeze(0),
                        SR
                    )

            if early_stop_triggered:
                pbar.close()
                print(f"[early-stop] Batch {batch_id} (oct={octave}, vel={velocity}) stopped at iter {iter_idx+1}: "
                      f"Δmean={mean_improve_pct:.2f}% Δvar={var_change_pct:.2f}% (window={es_window})")
                break

        all_batch_losses.append(batch_losses)
        batch_final_preds.append(y_pred.detach().cpu() if y_pred is not None else torch.empty(0))
        batch_final_targets.append(x_tgt.detach().cpu())

    return all_batch_losses, batch_final_preds, batch_final_targets


# -----------------------------
# Evaluation helpers
# -----------------------------
def per_sample_normalized_losses(
    pred_batches: List[torch.Tensor],
    target_batches: List[torch.Tensor],
    loss_fn: LogMSSLoss,
    device: torch.device,
    dtype: torch.dtype,
) -> List[float]:
    """Compute normalized MSS loss per sample across all batches."""
    out: List[float] = []
    for y_pred_batch, x_tgt_batch in zip(pred_batches, target_batches):
        if y_pred_batch.numel() == 0 or x_tgt_batch.numel() == 0:
            continue
        y_pred_batch = y_pred_batch.to(device=device, dtype=dtype)
        x_tgt_batch = x_tgt_batch.to(device=device, dtype=dtype)
        B = y_pred_batch.shape[0]
        for b in range(B):
            yb = y_pred_batch[b:b+1].unsqueeze(1)  # [1,1,T]
            xb = x_tgt_batch[b:b+1].unsqueeze(1)   # [1,1,T]
            norm_loss = loss_fn(yb, xb, normalize=True)  # normalized *evaluation* loss
            out.append(float(norm_loss.item()))
    return out


def per_oct_per_vel_table(mean_map: Dict[int, Dict[str, float]],
                   var_map: Dict[int, Dict[str, float]]) -> pd.DataFrame:
    """Format 'mean ± var' strings and append rightmost/ bottom aggregations."""
    df_mean = pd.DataFrame(mean_map).T[VEL_LABELS]
    df_var = pd.DataFrame(var_map).T[VEL_LABELS]

    # Human-friendly strings per cell
    df_table = pd.DataFrame(index=df_mean.index, columns=df_mean.columns)
    for oct_ in df_mean.index:
        for vel_ in df_mean.columns:
            m, v = df_mean.at[oct_, vel_], df_var.at[oct_, vel_]
            df_table.at[oct_, vel_] = "nan ± nan" if pd.isna(m) or pd.isna(v) else f"{m:.2f} ± {v:.2f}"

    # Rightmost column: mean across velocities per octave
    mean_across_vel = df_mean.mean(axis=1)
    var_across_vel = df_var.mean(axis=1)
    df_table["mean"] = [
        "nan ± nan" if pd.isna(mean_across_vel[oct_]) or pd.isna(var_across_vel[oct_])
        else f"{mean_across_vel[oct_]:.2f} ± {var_across_vel[oct_]:.2f}"
        for oct_ in df_mean.index
    ]

    # Bottom row: mean across octaves per velocity + overall cell
    mean_across_oct = df_mean.mean(axis=0)
    var_across_oct = df_var.mean(axis=0)
    bottom = [
        "nan ± nan" if pd.isna(mean_across_oct[vel_]) or pd.isna(var_across_oct[vel_])
        else f"{mean_across_oct[vel_]:.2f} ± {var_across_oct[vel_]:.2f}"
        for vel_ in df_mean.columns
    ]

    # Bottom-right overall
    overall_mean_vals = df_mean.values.flatten()
    overall_var_vals = df_var.values.flatten()
    overall_mean_vals = overall_mean_vals[~pd.isna(overall_mean_vals)]
    overall_var_vals = overall_var_vals[~pd.isna(overall_var_vals)]
    if len(overall_mean_vals) == 0 or len(overall_var_vals) == 0:
        overall_str = "nan ± nan"
    else:
        overall_str = f"{overall_mean_vals.mean():.2f} ± {overall_var_vals.mean():.2f}"
    bottom.append(overall_str)

    df_table.loc["mean"] = bottom
    return df_table


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    make_deterministic(args.seed)
    device = get_device()
    dtype = torch.float32

    # Fixed choice: evaluate all octaves & velocities
    octaves = [1, 2, 3, 4]
    velocities = VEL_LABELS

    # Dataset
    ds = GuitarAcousticDataset(root=ROOT, split=SPLIT)

    # Model & loss
    model = TimbreTransferKS(device=device, dtype=dtype, sr=SR).to(device)
    loss_fn = LogMSSLoss().to(device)

    # Accumulate results per (octave, velocity)
    per_combo_records: List[Tuple[int, str, List[List[float]], List[torch.Tensor], List[torch.Tensor]]] = []

    # Loop all combinations
    for octave in octaves:
        for velocity in velocities:
            keep_idx = indices_for_octave_velocity(ds, octave, velocity)
            print(f"Processing octave={octave} and velocity='{velocity}': matched {len(keep_idx)} items.")
            if len(keep_idx) == 0:
                continue

            ds_filt = Subset(ds, keep_idx)
            gen = torch.Generator()
            gen.manual_seed(args.seed)
            loader = DataLoader(
                ds_filt,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_examples,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=gen,
            )

            # Optionally limit batches for quick tests
            if args.num_batches is not None and args.num_batches > 0:
                # Create a thin wrapper that truncates iteration
                def limited_iter(dl, limit):
                    for i, item in enumerate(dl, start=1):
                        if i > limit:
                            break
                        yield item
                truncated_loader = limited_iter(loader, args.num_batches)
                # Run optimization on truncated iterator
                all_losses, preds, tgts = run_optim_over_loader(
                    loader=truncated_loader,  # type: ignore[arg-type]
                    model=model,
                    loss_fn=loss_fn,
                    num_frames=args.num_frames,
                    num_iters=args.num_iters,
                    lr=args.learning_rate,
                    device=device,
                    dtype=dtype,
                    octave=octave,
                    velocity=velocity,
                    es_window=args.es_window,
                    es_mean_pct=args.es_mean_pct,
                    es_var_pct=args.es_var_pct,
                    min_iters=args.min_iters,
                )
            else:
                all_losses, preds, tgts = run_optim_over_loader(
                    loader=loader,
                    model=model,
                    loss_fn=loss_fn,
                    num_frames=args.num_frames,
                    num_iters=args.num_iters,
                    lr=args.learning_rate,
                    device=device,
                    dtype=dtype,
                    octave=octave,
                    velocity=velocity,
                    es_window=args.es_window,
                    es_mean_pct=args.es_mean_pct,
                    es_var_pct=args.es_var_pct,
                    min_iters=args.min_iters,
                )

            per_combo_records.append((octave, velocity, all_losses, preds, tgts))

    print("Finished optimisation over all filtered dataset combinations.")

    # -----------------------------
    # Normalized evaluation → mean/variance per cell
    # -----------------------------
    results_mean: Dict[int, Dict[str, float]] = {o: {} for o in octaves}
    results_var: Dict[int, Dict[str, float]] = {o: {} for o in octaves}

    for octave in octaves:
        for velocity in velocities:
            # find the one record for this combo (if exists)
            recs = [r for r in per_combo_records if r[0] == octave and r[1] == velocity]
            if not recs:
                results_mean[octave][velocity] = float("nan")
                results_var[octave][velocity] = float("nan")
                continue

            _, _, _batch_losses, pred_batches, target_batches = recs[0]
            sample_losses = per_sample_normalized_losses(
                pred_batches, target_batches, loss_fn, device, dtype
            )
            if len(sample_losses) == 0:
                results_mean[octave][velocity] = float("nan")
                results_var[octave][velocity] = float("nan")
            else:
                t = torch.tensor(sample_losses)
                results_mean[octave][velocity] = float(t.mean())
                results_var[octave][velocity] = float(t.var(unbiased=False)) if t.numel() > 1 else 0.0

    # Format the combined table and save
    df_table = per_oct_per_vel_table(results_mean, results_var)
    EXPERIMENTS_RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(EXPERIMENTS_RESULTS_TABLES / "grad_per_oct_per_vel.csv")

    print("Combined mean ± variance loss per (octave, velocity):")
    print(df_table)


if __name__ == "__main__":
    main()