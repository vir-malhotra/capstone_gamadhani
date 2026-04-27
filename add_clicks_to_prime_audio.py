"""
Add GT beat click tracks to outputs_prime_audio/ WAV files.

For each synthesized WAV, overlays:
  - Loud low click at sam (cycle boundary)
  - Mid click at every GT beat
  - Soft high click at beat subdivisions (2x — useful for vilambit)

This lets us hear: does the generated pitch align with the GT beat structure,
or does it align at subdivisions (double tempo) in vilambit sections?
"""

import os
import numpy as np
import torchaudio
import torch
from scipy.signal import find_peaks

BEATS_DIR     = "/home/vm2426/HMR_processed/beats"
PRIME_AUDIO   = "outputs_prime_audio"
FPS           = 100
AUDIO_SR      = 16000

CLICK_DUR  = 0.025
SAM_FREQ   = 400.0;  SAM_GAIN  = 1.20
BEAT_FREQ  = 900.0;  BEAT_GAIN = 0.90
SUB2_FREQ  = 1400.0; SUB2_GAIN = 0.55   # 2x subdivision (half-beat)
SUB4_FREQ  = 2000.0; SUB4_GAIN = 0.30   # 4x subdivision (quarter-beat)


def make_click(freq, gain):
    n = int(AUDIO_SR * CLICK_DUR)
    t = np.linspace(0, CLICK_DUR, n, endpoint=False)
    c = np.sin(2 * np.pi * freq * t) * np.exp(-t / (CLICK_DUR * 0.25))
    return (c * gain).astype(np.float32)


def overlay(audio, pos_sample, click):
    end = min(pos_sample + len(click), len(audio))
    n = end - pos_sample
    if n > 0:
        audio[pos_sample:end] += click[:n]


def add_clicks_to_wav(wav_path, beat_arr, sam_arr, start_frame, out_path):
    waveform, sr = torchaudio.load(wav_path)
    if sr != AUDIO_SR:
        waveform = torchaudio.functional.resample(waveform, sr, AUDIO_SR)
    audio = waveform.mean(0).numpy().copy()

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.4  # headroom

    SEQ_LEN = 1200
    b_win = beat_arr[start_frame: start_frame + SEQ_LEN]
    s_win = sam_arr[start_frame:  start_frame + SEQ_LEN]

    beat_frames, _ = find_peaks(b_win, height=0.5, distance=3)
    sam_frames,  _ = find_peaks(s_win, height=0.5, distance=3)

    if len(beat_frames) < 2:
        return False

    beat_interval = np.median(np.diff(beat_frames))

    sam_click  = make_click(SAM_FREQ,  SAM_GAIN)
    beat_click = make_click(BEAT_FREQ, BEAT_GAIN)
    sub2_click = make_click(SUB2_FREQ, SUB2_GAIN)
    sub4_click = make_click(SUB4_FREQ, SUB4_GAIN)

    # 4x subdivisions (quarter-beats) — softest, highest pitch
    for bf in beat_frames:
        for k in [1, 3]:
            pos = int(bf + k * beat_interval / 4)
            if 0 <= pos < SEQ_LEN:
                overlay(audio, int(pos * AUDIO_SR / FPS), sub4_click)

    # 2x subdivisions (half-beats)
    for bf in beat_frames:
        pos = int(bf + beat_interval / 2)
        if 0 <= pos < SEQ_LEN:
            overlay(audio, int(pos * AUDIO_SR / FPS), sub2_click)

    # Beats
    for bf in beat_frames:
        overlay(audio, int(bf * AUDIO_SR / FPS), beat_click)

    # Sams — loudest, lowest pitch
    for sf in sam_frames:
        overlay(audio, int(sf * AUDIO_SR / FPS), sam_click)

    audio = np.clip(audio, -1.0, 1.0)
    torchaudio.save(out_path, torch.from_numpy(audio).unsqueeze(0), AUDIO_SR)
    return True


def main():
    folders = sorted(os.listdir(PRIME_AUDIO))
    print(f"Processing {len(folders)} folders in {PRIME_AUDIO}/\n")

    for folder in folders:
        uid = folder.split("_")[0]
        parts = folder.split("_")
        try:
            start = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        except ValueError:
            start = 0

        beat_path = os.path.join(BEATS_DIR, f"{uid}_beat.npy")
        if not os.path.exists(beat_path):
            print(f"  SKIP {folder} — no beat file")
            continue

        b = np.load(beat_path)
        beat_arr = b[0]
        sam_arr  = b[1]

        folder_path = os.path.join(PRIME_AUDIO, folder)
        wavs = [f for f in os.listdir(folder_path) if f.endswith(".wav") and "_click" not in f]

        n_done = 0
        for wav_name in wavs:
            wav_path = os.path.join(folder_path, wav_name)
            out_name = wav_name.replace(".wav", "_click.wav")
            out_path = os.path.join(folder_path, out_name)

            if os.path.exists(out_path):
                n_done += 1
                continue

            ok = add_clicks_to_wav(wav_path, beat_arr, sam_arr, start, out_path)
            if ok:
                n_done += 1

        print(f"  {folder}: {n_done}/{len(wavs)} click files ready")

    print("\nDone. Each *_click.wav has: low=sam, mid=beat, high=subdivision (2x).")


if __name__ == "__main__":
    main()
