import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchaudio
import pygad

from ddsp.timbre_transfer_ks import TimbreTransferKS
from ddsp.metrics import LogMSSLoss
from data.preprocess_subset import GuitarAcousticDataset

from experiments.gradient_opt import (
    SR, SPLIT, ROOT, VEL_LABELS,
    make_deterministic, seed_worker, get_device,
    per_sample_normalized_losses, per_oct_per_vel_table, collate_examples, indices_for_octave_velocity
)

from dirs import EXPERIMENTS_AUDIO_GENETIC, EXPERIMENTS_RESULTS_TABLES

GENETIC_DIR = EXPERIMENTS_AUDIO_GENETIC
GENETIC_DIR.mkdir(parents=True, exist_ok=True)

def parse_args():
    # Reduced default parameters for lighter local testing runs
    p = argparse.ArgumentParser(description="Genetic optimisation for TimbreTransferKS (6 params).")

    # data/optim shape (kept from gradient script)
    p.add_argument("--batch_size", type=int, default=1, help="Batch size")
    p.add_argument("--num_frames", type=int, default=16, help="Frames per clip (F)")

    # GA hyperparams
    p.add_argument("--ga_generations", type=int, default=40, help="Number of GA generations")
    p.add_argument("--ga_popsize", type=int, default=4, help="Population size (solutions per pop)")
    p.add_argument("--ga_parents_mating", type=int, default=2, help="#parents selected for mating per generation")
    p.add_argument("--ga_init_low", type=float, default=-1.0, help="Initial gene min value")
    p.add_argument("--ga_init_high", type=float, default=1.0, help="Initial gene max value")
    p.add_argument("--ga_mutation_prob", type=float, default=0.5, help="Per-gene mutation probability")
    p.add_argument("--ga_mutation_sigma", type=float, default=1.0, help="Random mutation range (+/-)")
    p.add_argument("--ga_keep_elite", type=int, default=1, help="Number of elitism solutions to keep")

    p.add_argument("--max_samples", type=int, default=None,
                   help="Crop audio to at most this many samples per item (None or <=0 to disable)")

    p.add_argument("--num_batches", type=int, default=None,
                   help="Max batches to run per (octave, velocity) combo (for quick tests)")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    return p.parse_args()




# -----------------------------
# GA optimisation core
# -----------------------------
def _flatten_params_shape(B: int, F: int, D: int = 6) -> int:
    return B * F * D


def _solution_to_params(
    sol: np.ndarray, B: int, F: int, D: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Map 1D genome → [B, F, D] tensor on device."""
    arr = sol.reshape(B, F, D).astype(np.float32 if dtype == torch.float32 else np.float64, copy=False)
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def run_ga_over_loader(
    loader: DataLoader,
    model: TimbreTransferKS,
    loss_fn: LogMSSLoss,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    octave: int,
    velocity: str,
    *,
    ga_generations: int,
    ga_popsize: int,
    ga_parents_mating: int,
    ga_init_low: float,
    ga_init_high: float,
    ga_mutation_prob: float,
    ga_mutation_sigma: float,
    ga_keep_elite: int,
    max_samples: int | None,
    ga_random_seed: int,
) -> Tuple[List[List[float]], List[torch.Tensor], List[torch.Tensor]]:
    """
    GA optimisation: return per-batch fitness curves (best per gen),
    final predictions (CPU), and targets (CPU) for downstream evaluation.
    """
    all_batch_bestfits: List[List[float]] = []
    batch_final_preds: List[torch.Tensor] = []
    batch_final_targets: List[torch.Tensor] = []
    sample_idx = 0

    for batch_id, (x_tgt, f0, onsets, bursts) in enumerate(loader, start=1):
        x_tgt = x_tgt.to(device=device, dtype=dtype)
        f0 = f0.to(device=device, dtype=dtype)
        onsets = onsets.to(device=device, dtype=torch.long)
        bursts = bursts.to(device=device, dtype=dtype)

        B, T = x_tgt.shape
        assert B == 1, "genetic_opt expects batch_size=1; got B={}".format(B)

        # Optional crop to cap runtime (helps on CPU nodes and laptops).
        if max_samples is not None and max_samples > 0 and T > max_samples:
            x_tgt = x_tgt[:, :max_samples]
            f0 = f0[:, :max_samples]
            bursts = bursts[:, :max_samples]
            # Re-pack onsets so that indices outside the new T are dropped.
            new_onsets = torch.full_like(onsets, -1)
            for i in range(onsets.size(0)):
                vals = onsets[i]
                vals = vals[vals >= 0]
                vals = vals[vals < max_samples]
                if vals.numel() > 0:
                    new_onsets[i, :vals.numel()] = vals
            onsets = new_onsets
            T = max_samples

        D = 6
        genome_len = _flatten_params_shape(B, num_frames, D)

        best_fitness_curve: List[float] = []

        def fitness_func(ga_instance, solution, solution_idx):
            # shape back to [B,F,6] → interpolate to [B,T,6] → forward → get loss → return NEGATIVE (maximize fitness)
            params_frames = _solution_to_params(solution, B, num_frames, D, device, dtype)  # [B,F,6]
            params_samples = F.interpolate(
                params_frames.permute(0, 2, 1),  # [B,6,F]
                size=T, mode="linear", align_corners=True
            ).permute(0, 2, 1)  # [B,T,6]

            with torch.no_grad():
                y_pred = model(params=params_samples, f0=f0, onsets=onsets, bursts=bursts)  # [B,T]
                loss = loss_fn(y_pred.unsqueeze(1), x_tgt.unsqueeze(1))  # scalar
                fitness = -float(loss.item())
            return fitness

        def on_gen(ga_instance):
            best_fitness = ga_instance.best_solution()[1]
            best_fitness_curve.append(best_fitness)
            # Print every generation; cheap and makes progress visible on clusters.
            print(f"[GA] batch={batch_id} gen={ga_instance.generations_completed} "
                  f"best_fitness={best_fitness:.6f}", flush=True)

        # Warm-up one forward to trigger any JIT/caches (Numba/torchlpc) and avoid "first-call" stalls.
        with torch.no_grad():
            _warm_params = torch.zeros(B, num_frames, D, device=device, dtype=dtype)
            _warm_samples = F.interpolate(_warm_params.permute(0, 2, 1), size=T, mode="linear", align_corners=True).permute(0, 2, 1)
            _ = model(params=_warm_samples, f0=f0, onsets=onsets, bursts=bursts)

        print(f"[GA] Starting: batch={batch_id} (oct={octave}, vel={velocity}), "
              f"T={T}, genome_len={genome_len}, pop={ga_popsize}, gens={ga_generations}", flush=True)

        ga = pygad.GA(
            num_generations=ga_generations,
            num_parents_mating=ga_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=ga_popsize,
            num_genes=genome_len,
            init_range_low=ga_init_low,
            init_range_high=ga_init_high,
            parent_selection_type="sss",
            keep_parents=0,
            keep_elitism=ga_keep_elite,
            crossover_type="single_point",
            mutation_type="random",
            random_mutation_min_val=-ga_mutation_sigma,
            random_mutation_max_val=+ga_mutation_sigma,
            mutation_probability=ga_mutation_prob,
            gene_type=np.float32 if dtype == torch.float32 else np.float64,
            on_generation=on_gen,
            stop_criteria=["saturate_10"],
            random_seed=ga_random_seed,
        )

        # Run GA
        ga.run()

        print(f"[GA] Completed batch={batch_id} (oct={octave}, vel={velocity}). "
              f"gens={ga.generations_completed} best={ga.best_solution()[1]:.6f}", flush=True)

        # Best solution → synthesize final prediction and SAVE (no targets saved)
        best_sol, best_fit, _ = ga.best_solution()
        with torch.no_grad():
            params_frames_best = _solution_to_params(best_sol, B, num_frames, D, device, dtype)
            params_samples_best = F.interpolate(
                params_frames_best.permute(0, 2, 1),
                size=T, mode="linear", align_corners=True
            ).permute(0, 2, 1)
            y_pred_best = model(params=params_samples_best, f0=f0, onsets=onsets, bursts=bursts)  # [B,T]

        # Save reconstructed audio ONLY, one sample at a time with a running index
        torchaudio.save(
            str(GENETIC_DIR / f"oct{octave}_vel{velocity}_{sample_idx}.wav"),
            y_pred_best[0].detach().cpu().unsqueeze(0),
            SR,
        )
        sample_idx += 1

        # Bookkeeping for downstream eval table
        all_batch_bestfits.append(best_fitness_curve)
        batch_final_preds.append(y_pred_best.detach().cpu())
        batch_final_targets.append(x_tgt.detach().cpu())

    return all_batch_bestfits, batch_final_preds, batch_final_targets


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

    per_combo_records: List[Tuple[int, str, List[List[float]], List[torch.Tensor], List[torch.Tensor]]] = []

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
                batch_size=1,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_examples,
                drop_last=False,
                worker_init_fn=seed_worker,
                generator=gen,
            )

            # Optional truncation for quick tests
            if args.num_batches is not None and args.num_batches > 0:
                def limited_iter(dl, limit):
                    for i, item in enumerate(dl, start=1):
                        if i > limit:
                            break
                        yield item
                truncated_loader = limited_iter(loader, args.num_batches)
                all_bestfits, preds, tgts = run_ga_over_loader(
                    loader=truncated_loader,  # type: ignore[arg-type]
                    model=model,
                    loss_fn=loss_fn,
                    num_frames=args.num_frames,
                    device=device,
                    dtype=dtype,
                    octave=octave,
                    velocity=velocity,
                    ga_generations=args.ga_generations,
                    ga_popsize=args.ga_popsize,
                    ga_parents_mating=args.ga_parents_mating,
                    ga_init_low=args.ga_init_low,
                    ga_init_high=args.ga_init_high,
                    ga_mutation_prob=args.ga_mutation_prob,
                    ga_mutation_sigma=args.ga_mutation_sigma,
                    ga_keep_elite=args.ga_keep_elite,
                    max_samples=args.max_samples,
                    ga_random_seed=args.seed,
                )
            else:
                all_bestfits, preds, tgts = run_ga_over_loader(
                    loader=loader,
                    model=model,
                    loss_fn=loss_fn,
                    num_frames=args.num_frames,
                    device=device,
                    dtype=dtype,
                    octave=octave,
                    velocity=velocity,
                    ga_generations=args.ga_generations,
                    ga_popsize=args.ga_popsize,
                    ga_parents_mating=args.ga_parents_mating,
                    ga_init_low=args.ga_init_low,
                    ga_init_high=args.ga_init_high,
                    ga_mutation_prob=args.ga_mutation_prob,
                    ga_mutation_sigma=args.ga_mutation_sigma,
                    ga_keep_elite=args.ga_keep_elite,
                    max_samples=args.max_samples,
                    ga_random_seed=args.seed,
                )

            per_combo_records.append((octave, velocity, all_bestfits, preds, tgts))

    print("Finished GA optimisation over all filtered dataset combinations.")

    # -----------------------------
    # Normalized evaluation → mean/variance per cell
    # -----------------------------
    results_mean: Dict[int, Dict[str, float]] = {o: {} for o in octaves}
    results_var: Dict[int, Dict[str, float]] = {o: {} for o in octaves}

    for octave in octaves:
        for velocity in velocities:
            recs = [r for r in per_combo_records if r[0] == octave and r[1] == velocity]
            if not recs:
                results_mean[octave][velocity] = float("nan")
                results_var[octave][velocity] = float("nan")
                continue

            _, _, _fits, pred_batches, target_batches = recs[0]
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

    # Save table
    df_table = per_oct_per_vel_table(results_mean, results_var)
    EXPERIMENTS_RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    df_table.to_csv(EXPERIMENTS_RESULTS_TABLES / "gene_per_oct_per_vel.csv")

    print("GA (genetic) mean ± variance loss per (octave, velocity):")
    print(df_table)


if __name__ == "__main__":
    main()