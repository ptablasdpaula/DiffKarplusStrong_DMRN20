from pathlib import Path
import csv
import statistics

import torch
from ddsp.metrics import MFCCDistance
import dirs

SR = 16000

def list_audio_files(folder: Path):
    return sorted(folder.rglob("*.wav"))

def load_mono_resampled(path: Path, sr: int):
    import torchaudio
    wav, file_sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0), file_sr

def compute_distances_and_save_csv(grad_dir: Path, targ_dir: Path, out_csv: Path, label: str):
    grad_files = list_audio_files(grad_dir)
    targ_files = list_audio_files(targ_dir)

    targ_by_name = {p.name: p for p in targ_files}

    pairs = [(g, targ_by_name[g.name]) for g in grad_files if g.name in targ_by_name]

    metric = MFCCDistance(
        sr=SR,
    )

    distances = []

    with torch.no_grad():
        for g_path, t_path in pairs:
            g, _ = load_mono_resampled(g_path, SR)
            t, _ = load_mono_resampled(t_path, SR)

            g_in = g.unsqueeze(0).unsqueeze(0).float()
            t_in = t.unsqueeze(0).unsqueeze(0).float()

            d = metric(g_in, t_in).item()
            distances.append((g_path.name, d))
            print(f"{label} - {g_path.name}: {d:.6f}")

    values = [d for _, d in distances]
    mean_val = statistics.fmean(values)
    var_val = statistics.pvariance(values)

    print(f"\n--- SUMMARY for {label} ---")
    print(f"Pairs: {len(values)}")
    print(f"Mean MFCC distance: {mean_val:.6f}")
    print(f"Variance: {var_val:.6f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "mfcc_distance"])
        for name, d in distances:
            writer.writerow([name, f"{d:.9f}"])
        writer.writerow([])
        writer.writerow(["mean", f"{mean_val:.9f}"])
        writer.writerow(["variance", f"{var_val:.9f}"])

    print(f"\nSaved per-file distances and summary to: {out_csv}\n")


def main():
    grad_dir: Path = dirs.EXPERIMENTS_AUDIO_GRADIENT
    genetic_dir: Path = getattr(dirs, "EXPERIMENTS_AUDIO_GENETIC", None)
    targ_dir: Path = dirs.EXPERIMENTS_AUDIO_TARGET

    out_dir: Path = getattr(dirs, "EXPERIMENTS_RESULTS_TABLES", Path(__file__).resolve().parent)

    if grad_dir is not None:
        compute_distances_and_save_csv(
            grad_dir,
            targ_dir,
            out_dir / "mfcc_distances_gradient.csv",
            label="Gradient vs Target"
        )

    if genetic_dir is not None:
        compute_distances_and_save_csv(
            genetic_dir,
            targ_dir,
            out_dir / "mfcc_distances_genetic.csv",
            label="Genetic vs Target"
        )


if __name__ == "__main__":
    main()
