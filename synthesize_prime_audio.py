"""
Convert outputs_prime/ pitch .npy files to audio using Stage 2 model.
Targets drut-laya val recordings (strongest beat conditioning).
"""

import copy, os, sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
import gamadhani.utils.pitch_to_audio_utils as p2a

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
AUDIO_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-model.ckpt"
AUDIO_QT   = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-qt.joblib"
PITCH_CFG  = "configs/diffusion_pitch_config.gin"
AUDIO_CFG  = "configs/pitch_to_audio_config.gin"
PRIME_DIR  = "outputs_prime"
AUDIO_DIR  = "outputs_prime_audio"

NUM_STEPS     = 100
AUDIO_SEQ_LEN = 750
SINGER_ID     = 3
SAMPLE_RATE   = 16000

# Drut + drut/madhya laya val recordings (one window each — first window)
TARGET_FOLDERS = [
    "20045_0",   # Raga Bilaskhani Todi — Drut
    "21040_0",   # Raga Marwa — Drut/Madhya
    "21055_0",   # Raag Bahar — Drut
    "22017_0",   # Raga Nat Narayan — Drut
    "22019_0",   # Raga Darbari Kanada — Drut
    "23020_0",   # Raga Bhageshri — Drut
    # Vilambit — to check if model adds beats at subdivisions
    "20002_0",   # Vilambit
    "20002_1200",# Vilambit (second window)
    "20011_0",   # Vilambit
    "20011_1200",# Vilambit (second window)
    "21012_0",   # Vilambit
    "21019_0",   # Vilambit
    "21020_0",   # Vilambit
    # Madhya
    "20016_0",   # Madhya
    "20016_1200",# Madhya (second window)
]

# Which pitch files to convert (key comparisons)
PITCH_FILES = [
    "pitch_gt.npy",           # ground truth pitch
    "pitch_noprime_gt.npy",   # generated, no prime, GT beat
    "pitch_prime400_gt.npy",  # pitch prime 400fr, GT beat
    "pitch_prime400_shuf.npy",# pitch prime 400fr, shuffled beat
    "pitch_beatprime400_gt.npy",  # beat prime 400fr, GT beat
    "pitch_beatprime400_shuf.npy",# beat prime 400fr, shuffled beat
]


def qt_to_tokens(pitch_qt_space: np.ndarray, qt, min_clip: int = 200) -> torch.Tensor:
    tokens = qt.inverse_transform(pitch_qt_space.reshape(-1, 1))
    tokens = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1)
    tokens = torch.round(tokens)
    tokens[tokens < min_clip] = float('nan')
    return tokens


def pitch_to_audio(pitch_npy, qt, audio_model, invert_audio_fn, device):
    tokens = qt_to_tokens(pitch_npy, qt).to(device)
    interp = p2a.interpolate_pitch(tokens, AUDIO_SEQ_LEN)
    interp = torch.nan_to_num(interp, nan=196.0).squeeze(1).float()
    singer_tensor = torch.tensor([SINGER_ID]).to(device)
    with torch.no_grad():
        audio_samples, _, _ = audio_model.sample_cfg(
            1, f0=interp, num_steps=NUM_STEPS,
            singer=singer_tensor, strength=3,
            invert_audio_fn=None)
        audio_wav = invert_audio_fn(audio_samples)
    return audio_wav[0].unsqueeze(0).cpu()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load models
    pitch_model, pitch_qt, _, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=PITCH_CFG, device=device)
    qt = joblib.load(QT_PATH)

    audio_model, audio_qt, audio_seq_len_cfg, invert_audio_fn = load_audio_fns(
        audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
    audio_model.eval()
    print("Models loaded.")

    os.makedirs(AUDIO_DIR, exist_ok=True)

    total = len(TARGET_FOLDERS) * len(PITCH_FILES)
    done = 0

    for folder in TARGET_FOLDERS:
        src_dir = os.path.join(PRIME_DIR, folder)
        dst_dir = os.path.join(AUDIO_DIR, folder)
        os.makedirs(dst_dir, exist_ok=True)

        for pitch_file in PITCH_FILES:
            src = os.path.join(src_dir, pitch_file)
            if not os.path.exists(src):
                print(f"  SKIP (missing): {src}")
                continue

            wav_name = pitch_file.replace(".npy", ".wav")
            dst = os.path.join(dst_dir, wav_name)

            if os.path.exists(dst):
                print(f"  EXISTS: {folder}/{wav_name}")
                done += 1
                continue

            pitch_npy = np.load(src)
            print(f"  [{done+1}/{total}] {folder}/{wav_name} ...", flush=True)
            wav = pitch_to_audio(pitch_npy, qt, audio_model, invert_audio_fn, device)
            torchaudio.save(dst, wav, SAMPLE_RATE)
            done += 1

        print(f"  Done: {folder}")

    print(f"\nAll audio saved to {AUDIO_DIR}/")


if __name__ == "__main__":
    main()
