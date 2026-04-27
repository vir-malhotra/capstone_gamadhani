"""
Generate prime400_zero condition (pitch prime + zero beat conditioning) for
the 6 selected windows used in the comparison table.
"""
import copy, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import joblib
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset
import gamadhani.utils.pitch_to_audio_utils as p2a
from evaluate_beat_alignment import detect_onsets, beat_alignment_f1
from generate_cfg_with_prime import (
    BeatConditionedUNet, pitch_to_audio, add_clicks_to_wav,
    PITCH_PATH, QT_PATH, AUDIO_PATH, AUDIO_QT,
    PITCH_CFG, AUDIO_CFG, BEST_CKPT,
    NUM_STEPS, AUDIO_SEQ_LEN, PRIME_LEN, GUIDANCE_SCALE, SAMPLE_RATE, SINGER_ID
)

WINDOWS = ["21007_1200", "20002_0", "21017_1200", "20002_1200", "21017_0", "21007_0"]
OUT_DIR = "outputs_cfg_prime"
COND_NAME = "prime400_zero"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load models
pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
    pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
    config_path=PITCH_CFG, device=device)

beat_model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
ckpt = torch.load(BEST_CKPT, map_location=device)
beat_model.load_state_dict(ckpt["state_dict"])
beat_model.eval()
print(f"Loaded beat model  epoch={ckpt['epoch']}")

qt = joblib.load(QT_PATH)

audio_model, _, _, invert_audio_fn = load_audio_fns(
    audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
audio_model.eval()
print("Loaded audio model")

val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                        val_ratio=0.17, seed=42,
                        window_stride=1200, seq_len=1200)

# Build uid_start -> dataset index map
index_map = {}
for i in range(len(val_ds)):
    item = val_ds[i]
    key = f"{item['uid']}_{item['start']}"
    index_map[key] = i

for tag in WINDOWS:
    out = os.path.join(OUT_DIR, tag)
    npy_path  = os.path.join(out, f"pitch_{COND_NAME}.npy")
    wav_path  = os.path.join(out, f"audio_{COND_NAME}.wav")
    click_path = os.path.join(out, f"audio_{COND_NAME}_click.wav")

    if os.path.exists(click_path):
        print(f"[{tag}] EXISTS — skipping")
        continue

    if tag not in index_map:
        print(f"[{tag}] NOT FOUND in val dataset — skipping")
        continue

    idx  = index_map[tag]
    item = val_ds[idx]
    beat_np = item["beat"].numpy()
    beat_frames = np.where(beat_np > 0)[0]
    pitch_gt_np = item["normalized_pitch"].numpy()

    # Condition: pitch prime + zero beat
    beat_zero   = torch.zeros(1, 1, len(beat_np)).to(device)
    pitch_prime = torch.tensor(
        pitch_gt_np[:PRIME_LEN], dtype=torch.float32).reshape(1, 1, PRIME_LEN)

    print(f"[{tag}] Generating {COND_NAME} ...")
    with torch.no_grad():
        gen = beat_model.sample(beat_zero, guidance_scale=1.0,
                                prime=pitch_prime).squeeze().cpu().numpy()

    np.save(npy_path, gen)

    onsets = detect_onsets(gen, qt)
    m = beat_alignment_f1(onsets, beat_frames)
    print(f"  F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}")

    # Update metrics.json
    metrics_path = os.path.join(out, "metrics.json")
    existing = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    existing[COND_NAME] = m
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    # Audio synthesis
    print(f"  Synthesizing audio ...")
    wav = pitch_to_audio(gen, qt, audio_model, invert_audio_fn, device)
    torchaudio.save(wav_path, wav, SAMPLE_RATE)
    wav_click = add_clicks_to_wav(wav, beat_np, subdivisions=4)
    torchaudio.save(click_path, wav_click, SAMPLE_RATE)
    print(f"  Saved: {click_path}")

print("\nDone.")
