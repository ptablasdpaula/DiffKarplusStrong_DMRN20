from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from ddsp.utils import detect_f0, detect_onsets, bursts_at_onsets

SAMPLE_RATE = 16_000
SEGMENT_LENGTH = SAMPLE_RATE * 4          # 64_000 samples
HOP_F0 = 4096                             # YIN hop
E2_MIDI, E6_MIDI = 40, 88                 # E2..E6

@torch.no_grad()
def preprocess_nsynth_guitar_acoustic(
    nsynth_root: str = "nsynth",
    out_dir: str   = "nsynth_preprocessed",
    splits: List[str] = ("test",),
    seed: int = 42,
) -> None:
    script_dir = Path(__file__).resolve().parent
    nsynth_root = (script_dir / nsynth_root).resolve()
    out_dir = (script_dir / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n▶ Processing split: {split}")
        split_in = nsynth_root / f"nsynth-{split}"
        examples_json = split_in / "examples.json"
        assert examples_json.exists(), f"Missing {examples_json}"

        meta_json: Dict[str, Any] = json.loads(examples_json.read_text())

        # ---- Filter keys
        keys = []
        for k, m in meta_json.items():
            fam = m.get("instrument_family_str", "")
            src = m.get("instrument_source_str", "")
            midi = m.get("pitch", None)

            if fam != "guitar":                 # only guitar family
                continue
            if src != "acoustic":               # only acoustic source
                continue
            if midi is None or not (E2_MIDI <= midi <= E6_MIDI):  # E2..E6
                continue

            # Qualities: keep tempo-synced (index 8), but remove reverberation (index 9)
            q = m.get("qualities", [0]*10)
            has_reverb = (len(q) > 9 and q[9] == 1)
            if has_reverb:
                continue

            keys.append(k)

        print(f"Kept {len(keys)} items after filtering (guitar/acoustic/E2..E6/no-reverb).")

        # ---- Prepare output split dir
        split_out = out_dir / split
        (split_out / "items").mkdir(parents=True, exist_ok=True)  # per-item .pt tensors
        metadata: Dict[str, Any] = {}

        # ---- Iterate over items
        for k in tqdm(keys, ncols=100):
            wav_path = split_in / "audio" / f"{k}.wav"
            x, sr = torchaudio.load(str(wav_path))  # (1, T)
            assert sr == SAMPLE_RATE, f"{k}: expected {SAMPLE_RATE}, got {sr}"
            x = x.squeeze(0)  # (T,)
            assert x.numel() == SEGMENT_LENGTH, f"{k}: expected {SEGMENT_LENGTH} samples"

            # ---- Pitch (Hz) with librosa.yin → interpolate to sample rate
            x_np = x.detach().cpu().numpy()
            f0_frames = detect_f0(x_np, sr=SAMPLE_RATE, frame_length=SEGMENT_LENGTH, hop_length=SEGMENT_LENGTH)
            f0_hz_value = float(f0_frames.mean())
            f0_hz_sr = np.full_like(x_np, f0_hz_value, dtype=np.float32)

            # ---- Onsets (sample indices)
            onsets = detect_onsets(x_np, sr=SAMPLE_RATE).astype(np.int64)

            # ---- Bursts at onsets
            bursts = bursts_at_onsets(x_np, sr=SAMPLE_RATE, onset_samples=onsets, f0=f0_hz_sr, seed=seed).astype(np.float32)

            # ---- Save tensors
            item_pt = split_out / "items" / f"{k}.pt"
            torch.save({
                "audio":  x.cpu(),                              # (T,)
                "f0_hz":  torch.from_numpy(f0_hz_sr),           # (T,)
                "bursts": torch.from_numpy(bursts),             # (T,)
                "onsets": torch.from_numpy(onsets),             # (N_onsets,)
            }, item_pt)

            # ---- Write compact metadata entry
            m = meta_json[k]
            metadata[k] = {
                "path": f"{split}/items/{k}.pt",
                "num_samples": int(x.numel()),
                "instrument_family_str": m.get("instrument_family_str", ""),
                "instrument_source_str": m.get("instrument_source_str", ""),
                "midi_pitch": int(m.get("pitch", -1)),
                "midi_velocity": int(m.get("velocity", -1)),
                "onsets": onsets.tolist(),   # list[int]
            }

        # ---- Save split metadata
        (split_out / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Wrote {split_out/'metadata.json'} with {len(metadata)} items.")

# --------------------------
# PyTorch Dataset
# --------------------------
class GuitarAcousticDataset(torch.utils.data.Dataset):
    """
    Yields: audio (T,), f0_hz (T,), onsets (N_onsets,), bursts (T,)
    Only includes: guitar/acoustic, E2..E6, no-reverb (by construction of metadata).
    """
    def __init__(self, root: str = "nsynth_preprocessed", split: str = "test"):
        repo_root = Path(__file__).resolve().parent.parent
        data_root = repo_root / "data"
        self.base = data_root / Path(root) / split  # e.g., <repo>/data/nsynth_preprocessed/test
        self.meta = json.loads((self.base / "metadata.json").read_text())
        self.keys = list(self.meta.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[idx]
        item_path = self.base.parent / Path(self.meta[k]["path"])  # <repo>/data/nsynth_preprocessed/test/items/<file>.pt
        pt = torch.load(item_path, weights_only=True)

        audio  = pt["audio"].float()         # (T,)
        f0_hz  = pt["f0_hz"].float()         # (T,)
        bursts = pt["bursts"].float()        # (T,)
        onsets = pt["onsets"].long()         # (N_onsets,)

        return audio, f0_hz, onsets, bursts

    def get_filename(self, idx: int) -> str:
        return self.keys[idx]

# --------------------------
# Run it directly
# --------------------------
if __name__ == "__main__":
    preprocess_nsynth_guitar_acoustic(
        nsynth_root="nsynth",
        out_dir="nsynth_preprocessed",
    )

    # Test dataset loading
    ds = GuitarAcousticDataset("nsynth_preprocessed", split="test")
    print(f"Loaded {len(ds)} examples.")
    audio, f0, onsets, bursts = ds[0]
    print(audio.shape, f0.shape, onsets.shape, bursts.shape)
