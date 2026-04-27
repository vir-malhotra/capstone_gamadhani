"""
Add GT beat clicks (with subdivisions) to original GT vocal audio.

Produces 12s clips with:
  - Loud low click (400Hz) at every sam (cycle boundary)
  - Mid click (900Hz) at every GT beat
  - Softer click (1400Hz) at 2x subdivisions (half-beat)
  - Softest click (2000Hz) at 4x subdivisions (quarter-beat)

Click parameters match add_clicks_to_prime_audio.py for direct comparison.
"""

import numpy as np
import torchaudio
import torch
import os
from scipy.signal import find_peaks

VOCALS_DIR = "/home/vm2426/HMR_processed/vocals"
BEATS_DIR  = "/home/vm2426/HMR_processed/beats"
OUT_DIR    = "gt_click_sanity"

VOCAL_SR  = 16000
BEAT_SR   = 100
WINDOW_S  = 12

# Match add_clicks_to_prime_audio.py exactly
CLICK_DUR  = 0.025
SAM_FREQ   = 400.0;  SAM_GAIN  = 1.20
BEAT_FREQ  = 900.0;  BEAT_GAIN = 0.90
SUB2_FREQ  = 1400.0; SUB2_GAIN = 0.55
SUB4_FREQ  = 2000.0; SUB4_GAIN = 0.30

# All 15 folders from synthesize_prime_audio.py TARGET_FOLDERS
# start_s = start_frame / 100
TARGETS = [
    # Drut
    {"uid": "20045", "start_s": 0,  "label": "20045_0_drut"},
    {"uid": "21040", "start_s": 0,  "label": "21040_0_drut"},
    {"uid": "21055", "start_s": 0,  "label": "21055_0_drut"},
    {"uid": "22017", "start_s": 0,  "label": "22017_0_drut"},
    {"uid": "22019", "start_s": 0,  "label": "22019_0_drut"},
    {"uid": "23020", "start_s": 0,  "label": "23020_0_drut"},
    # Vilambit
    {"uid": "20002", "start_s": 0,  "label": "20002_0_vilambit"},
    {"uid": "20002", "start_s": 12, "label": "20002_1200_vilambit"},
    {"uid": "20011", "start_s": 0,  "label": "20011_0_vilambit"},
    {"uid": "20011", "start_s": 12, "label": "20011_1200_vilambit"},
    {"uid": "21012", "start_s": 0,  "label": "21012_0_vilambit"},
    {"uid": "21019", "start_s": 0,  "label": "21019_0_vilambit"},
    {"uid": "21020", "start_s": 0,  "label": "21020_0_vilambit"},
    # Madhya
    {"uid": "20016", "start_s": 0,  "label": "20016_0_madhya"},
    {"uid": "20016", "start_s": 12, "label": "20016_1200_madhya"},
]


def make_click(freq, gain):
    n = int(VOCAL_SR * CLICK_DUR)
    t = np.linspace(0, CLICK_DUR, n, endpoint=False)
    click = np.sin(2 * np.pi * freq * t) * np.exp(-t / (CLICK_DUR * 0.25))
    return (click * gain).astype(np.float32)


def overlay_click(audio, sample_pos, click):
    end = min(sample_pos + len(click), len(audio))
    n = end - sample_pos
    if n > 0:
        audio[sample_pos:end] += click[:n]


def process(uid, start_s, label):
    vocal_path = os.path.join(VOCALS_DIR, f"{uid}_vocals.wav")
    beat_path  = os.path.join(BEATS_DIR,  f"{uid}_beat.npy")

    if not os.path.exists(vocal_path):
        print(f"  SKIP {uid} — vocals not found")
        return
    if not os.path.exists(beat_path):
        print(f"  SKIP {uid} — beat not found")
        return

    waveform, sr = torchaudio.load(vocal_path)
    if sr != VOCAL_SR:
        waveform = torchaudio.functional.resample(waveform, sr, VOCAL_SR)
    audio = waveform.mean(0).numpy()

    start_sample = int(start_s * VOCAL_SR)
    end_sample   = start_sample + WINDOW_S * VOCAL_SR
    if end_sample > len(audio):
        print(f"  SKIP {uid} — audio too short for window at {start_s}s")
        return
    audio = audio[start_sample:end_sample].copy()

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.6

    beat_full = np.load(beat_path)  # (3, T)
    beat_ch = beat_full[0]
    sam_ch  = beat_full[1]

    start_bf = int(start_s * BEAT_SR)
    end_bf   = start_bf + WINDOW_S * BEAT_SR
    beat_window = beat_ch[start_bf:end_bf]
    sam_window  = sam_ch[start_bf:end_bf]

    beat_frames, _ = find_peaks(beat_window, height=0.5, distance=3)
    sam_frames,  _ = find_peaks(sam_window,  height=0.5, distance=3)

    if len(beat_frames) == 0:
        print(f"  WARNING {uid} — no beats detected in window")
        return

    beat_interval = np.median(np.diff(beat_frames)) if len(beat_frames) > 1 else None
    bpm = 60.0 / (float(beat_interval) / BEAT_SR) if beat_interval else 0

    print(f"  {label}: {len(beat_frames)} beats, {len(sam_frames)} sams, ~{bpm:.0f} BPM")

    sam_click  = make_click(SAM_FREQ,  SAM_GAIN)
    beat_click = make_click(BEAT_FREQ, BEAT_GAIN)
    sub2_click = make_click(SUB2_FREQ, SUB2_GAIN)
    sub4_click = make_click(SUB4_FREQ, SUB4_GAIN)

    mix = audio.copy()

    # 4x subdivisions (quarter-beat) — softest
    if beat_interval is not None:
        for bf in beat_frames:
            for k in [1, 3]:
                pos = int(bf + k * beat_interval / 4)
                if 0 <= pos < len(beat_window):
                    overlay_click(mix, int(pos * VOCAL_SR / BEAT_SR), sub4_click)

    # 2x subdivisions (half-beat)
    if beat_interval is not None:
        for bf in beat_frames:
            pos = int(bf + beat_interval / 2)
            if 0 <= pos < len(beat_window):
                overlay_click(mix, int(pos * VOCAL_SR / BEAT_SR), sub2_click)

    # Beats
    for bf in beat_frames:
        overlay_click(mix, int(bf * VOCAL_SR / BEAT_SR), beat_click)

    # Sams — loudest, lowest
    for sf in sam_frames:
        overlay_click(mix, int(sf * VOCAL_SR / BEAT_SR), sam_click)

    mix = np.clip(mix, -1.0, 1.0)
    out_path = os.path.join(OUT_DIR, f"{label}.wav")
    torchaudio.save(out_path, torch.from_numpy(mix).unsqueeze(0), VOCAL_SR)
    print(f"    → {out_path}")

    clean = np.clip(audio, -1.0, 1.0)
    clean_path = os.path.join(OUT_DIR, f"{label}_clean.wav")
    torchaudio.save(clean_path, torch.from_numpy(clean).unsqueeze(0), VOCAL_SR)
    print(f"    → {clean_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating GT click sanity clips → {OUT_DIR}/\n")
    print("Click key: 400Hz=sam, 900Hz=beat, 1400Hz=half-beat, 2000Hz=quarter-beat\n")

    for t in TARGETS:
        process(t["uid"], t["start_s"], t["label"])

    print(f"\nDone.")


if __name__ == "__main__":
    main()
