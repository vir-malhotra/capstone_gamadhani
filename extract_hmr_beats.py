"""
Run Beat Transformer on all usable HMR audio files to extract beat activations,
then convert to 100Hz beat signals matching the GT beat signal format.

Output per UID in HMR_processed/beats_extracted/:
  {uid}_beat.npy   — shape [3, N_frames_at_100Hz]
                     ch0: beat pulse  (smoothed, same as GT format)
                     ch1: downbeat pulse
                     ch2: cycle position (interpolated between downbeats)

Runs sequentially (Spleeter + Beat Transformer together OOM on GPU if parallel).
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d
from spleeter.separator import Separator

sys.path.insert(0, '/home/vm2426/beat_conditioning/Beat-Transformer/code')
from DilatedTransformer import Demixed_DilatedTransformerModel as BeatTransformer

AUDIO_DIR   = "/home/vm2426/HMR dataset/audio"
INSTR_CSV   = "/home/vm2426/beat_transformer_hindustani/inference/hmr_instruments.csv"
OUT_DIR     = "/home/vm2426/HMR_processed/beats_extracted"
MODEL_PATH  = "/home/vm2426/beat_conditioning/Beat-Transformer/checkpoint/fold_0_trf_param.pt"
DEVICE      = "cuda:1"
SR          = 44100
BT_FPS      = SR / 1024          # 43.07 — Beat Transformer output FPS
TARGET_FPS  = 100                 # output FPS (matches GT beats and pitch)

EXCLUDED = {'20028','20032','21018','21025','22011','23014','23015','23019','20052'}


def uid_to_audio_path(uid):
    for f in os.listdir(AUDIO_DIR):
        if f.split('_')[1] == uid:
            return os.path.join(AUDIO_DIR, f)
    return None


def make_beat_signal(beat_times, down_times, total_frames, fps=TARGET_FPS):
    """Convert beat/downbeat timestamps (seconds) to 3-channel 100Hz signal."""
    N = total_frames

    def pulse(times):
        arr = np.zeros(N, dtype=np.float32)
        for t in times:
            idx = int(round(t * fps))
            if 0 <= idx < N:
                arr[idx] = 1.0
        # smooth with 5-sample box (same as GT)
        arr = np.convolve(arr, np.ones(5), mode='same')
        return np.clip(arr, 0, 1)

    ch0 = pulse(beat_times)
    ch1 = pulse(down_times)

    # Cycle position: linearly interpolate 0→1 between consecutive downbeats
    ch2 = np.zeros(N, dtype=np.float32)
    down_frames = [int(round(t * fps)) for t in down_times if 0 <= int(round(t * fps)) < N]
    if len(down_frames) >= 2:
        for i in range(len(down_frames) - 1):
            s, e = down_frames[i], down_frames[i+1]
            if e > s:
                ch2[s:e] = np.linspace(0, 1, e - s, endpoint=False)
        ch2[down_frames[-1]:] = 0.0

    return np.stack([ch0, ch1, ch2], axis=0)   # [3, N]


def run_beat_transformer(audio_path, separator, model):
    """Run Spleeter + Beat Transformer on a single audio file."""
    audio, _ = librosa.load(audio_path, sr=SR, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[0] == 2:
        audio = audio.T   # [T, 2]

    duration = len(audio) / SR

    # Spleeter separation
    tmp = "/tmp/hmr_beat_tmp.wav"
    tmp_dir = "/tmp/hmr_beat_out"
    sf.write(tmp, audio, SR)
    separator.separate_to_file(tmp, tmp_dir)

    stems = [librosa.load(f"{tmp_dir}/hmr_beat_tmp/{s}.wav", sr=SR, mono=True)[0]
             for s in ['vocals', 'drums', 'bass', 'piano', 'other']]

    spectrograms = []
    for stem in stems:
        mel = librosa.feature.melspectrogram(y=stem, sr=SR, n_fft=4096,
                                             hop_length=1024, n_mels=128, fmax=11000)
        spectrograms.append(librosa.power_to_db(mel, ref=np.max))

    spec = np.transpose(np.stack(spectrograms, axis=0), (0, 2, 1))  # [5, T, 128]

    with torch.no_grad():
        pred, _ = model(torch.from_numpy(spec).float().unsqueeze(0).to(DEVICE))
        beat_act = torch.sigmoid(pred[0, :, 0]).cpu().numpy()
        down_act = torch.sigmoid(pred[0, :, 1]).cpu().numpy()

    return beat_act, down_act, duration


def activations_to_times(beat_act, down_act, fps=BT_FPS,
                          beat_height=0.3, down_height=0.3,
                          min_beat_interval=0.2, min_down_interval=2.0):
    """Peak-pick activations to get beat/downbeat timestamps."""
    beat_peaks, _ = signal.find_peaks(beat_act, height=beat_height,
                                       distance=int(min_beat_interval * fps))
    down_peaks, _ = signal.find_peaks(down_act, height=down_height,
                                       distance=int(min_down_interval * fps))
    return beat_peaks / fps, down_peaks / fps


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INSTR_CSV)
    df['UID'] = df['UID'].astype(str)
    usable_uids = [uid for uid in df[df['excluded'] != True]['UID']
                   if uid not in EXCLUDED]
    print(f"Processing {len(usable_uids)} UIDs...")

    # Load models once
    separator = Separator('spleeter:5stems')
    model = BeatTransformer(attn_len=5, instr=5, ntoken=2, dmodel=256,
                            nhead=8, d_hid=1024, nlayers=9, norm_first=True)
    ckpt = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    model.to(DEVICE).eval()
    print("Models loaded.")

    done, failed = 0, []
    for i, uid in enumerate(usable_uids):
        out_path = os.path.join(OUT_DIR, f"{uid}_beat.npy")
        if os.path.exists(out_path):
            print(f"  [{i+1}/{len(usable_uids)}] {uid} — skipping (exists)")
            done += 1
            continue

        audio_path = uid_to_audio_path(uid)
        if audio_path is None:
            print(f"  [{i+1}/{len(usable_uids)}] {uid} — no audio file!")
            failed.append(uid)
            continue

        try:
            beat_act, down_act, duration = run_beat_transformer(audio_path, separator, model)

            beat_times, down_times = activations_to_times(beat_act, down_act)

            total_frames = int(round(duration * TARGET_FPS))
            beat_sig = make_beat_signal(beat_times, down_times, total_frames)

            np.save(out_path, beat_sig)
            print(f"  [{i+1}/{len(usable_uids)}] {uid}  beats={len(beat_times)}  "
                  f"downs={len(down_times)}  shape={beat_sig.shape}", flush=True)
            done += 1
        except Exception as e:
            print(f"  [{i+1}/{len(usable_uids)}] {uid} FAILED: {e}")
            failed.append(uid)

    print(f"\nDone: {done}/{len(usable_uids)}")
    if failed:
        print(f"Failed UIDs: {failed}")


if __name__ == "__main__":
    main()
