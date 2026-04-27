"""
Add click tracks to generated audio outputs.

For each output folder, reads beat.npy (GT beat signal at 100Hz) and overlays
a click sound at each beat position onto audio_gt.wav, audio_shuf.wav, audio_zero.wav.

Outputs: audio_gt_click.wav, audio_shuf_click.wav, audio_zero_click.wav
"""

import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks

OUTPUTS_DIR = "outputs"
AUDIO_SR    = 16000   # Hz
BEAT_SR     = 100     # Hz
FRAMES_PER_BEAT_FRAME = AUDIO_SR // BEAT_SR  # 160 samples per beat frame

CLICK_FREQ     = 1000.0   # Hz — sine burst frequency
CLICK_DURATION = 0.020    # seconds
CLICK_GAIN     = 0.35     # relative to audio peak


def make_click(sr: int, freq: float, duration: float) -> np.ndarray:
    """Short sine burst with exponential decay."""
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    click = np.sin(2 * np.pi * freq * t)
    # exponential decay envelope
    click *= np.exp(-t / (duration * 0.3))
    return click.astype(np.float32)


def extract_beat_frames(beat: np.ndarray) -> np.ndarray:
    """Find peak positions (in beat-SR frames) from smoothed beat signal."""
    peaks, _ = find_peaks(beat, height=0.5, distance=5)
    return peaks


def add_clicks(audio_int16: np.ndarray, beat_frames: np.ndarray,
               click_template: np.ndarray, gain: float) -> np.ndarray:
    """Overlay click at each beat position. Returns int16."""
    audio = audio_int16.astype(np.float32)
    audio_peak = np.abs(audio).max()
    click = click_template * gain * audio_peak

    for bf in beat_frames:
        sample_pos = int(bf * FRAMES_PER_BEAT_FRAME)
        end = min(sample_pos + len(click), len(audio))
        clip_len = end - sample_pos
        if clip_len <= 0:
            continue
        audio[sample_pos:end] += click[:clip_len]

    # clip and convert back to int16
    audio = np.clip(audio, -32768, 32767)
    return audio.astype(np.int16)


def process_folder(folder_path: str, click_template: np.ndarray):
    beat_path = os.path.join(folder_path, "beat.npy")
    if not os.path.exists(beat_path):
        print(f"  SKIP {folder_path} — no beat.npy")
        return

    beat = np.load(beat_path)
    beat_frames = extract_beat_frames(beat)

    for condition in ["gt", "shuf", "zero"]:
        src = os.path.join(folder_path, f"audio_{condition}.wav")
        dst = os.path.join(folder_path, f"audio_{condition}_click.wav")
        if not os.path.exists(src):
            continue
        sr, audio = wavfile.read(src)
        assert sr == AUDIO_SR, f"Unexpected SR {sr} in {src}"
        out = add_clicks(audio, beat_frames, click_template, CLICK_GAIN)
        wavfile.write(dst, sr, out)

    print(f"  {os.path.basename(folder_path):15s}  {len(beat_frames)} beats")


def main():
    click_template = make_click(AUDIO_SR, CLICK_FREQ, CLICK_DURATION)

    folders = sorted([
        os.path.join(OUTPUTS_DIR, d)
        for d in os.listdir(OUTPUTS_DIR)
        if os.path.isdir(os.path.join(OUTPUTS_DIR, d))
    ])

    print(f"Processing {len(folders)} folders...\n")
    for folder in folders:
        process_folder(folder, click_template)

    print(f"\nDone. Click-track files saved as audio_*_click.wav in each folder.")


if __name__ == "__main__":
    main()
