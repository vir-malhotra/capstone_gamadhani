"""
Generate pitch + audio under all conditions using the CFG model with prime support.

Conditions per window:
  noprime_gt_gs3    — GT beat, no prime, CFG gs=3  (same as existing outputs_cfg)
  noprime_shuf_gs3  — shuffled beat, no prime, CFG gs=3
  noprime_zero      — zero beat, unconditioned
  prime400_gt_gs3   — pitch prime 400fr, GT beat, CFG gs=3
  prime400_shuf_gs3 — pitch prime 400fr, shuffled beat, CFG gs=3
  beatprime400_gt_gs3   — beat prime 400fr, GT beat, CFG gs=3
  beatprime400_shuf_gs3 — beat prime 400fr, shuffled beat, CFG gs=3

Saves to outputs_cfg_prime/{uid}_{start}/:
  beat.npy
  pitch_{cond}.npy  for each condition
  audio_{cond}.wav  + audio_{cond}_click.wav
  plot.png
  metrics.json
"""

import argparse, copy, json, os, sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torchaudio
import joblib
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset
import gamadhani.utils.pitch_to_audio_utils as p2a
from evaluate_beat_alignment import detect_onsets, beat_alignment_f1

PITCH_PATH  = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH     = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
AUDIO_PATH  = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-model.ckpt"
AUDIO_QT    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-qt.joblib"
PITCH_CFG   = "configs/diffusion_pitch_config.gin"
AUDIO_CFG   = "configs/pitch_to_audio_config.gin"
BEST_CKPT   = "checkpoints/hmr_gt_beats_cfg/best.ckpt"

NUM_STEPS     = 100
AUDIO_SEQ_LEN = 750
SINGER_ID     = 3
SAMPLE_RATE   = 16000
PRIME_LEN     = 400
GUIDANCE_SCALE = 3.0
SILENCE_QT    = -1.077
NOTE_QT       = 0.2
NOTE_DUR      = 10

CLICK_FREQ     = 1000.0
CLICK_DURATION = 0.020
CLICK_GAIN     = 0.50
SUBCLICK_GAIN  = 0.20
SUBCLICK_FREQ  = 600.0


def make_click(freq, duration, sr=SAMPLE_RATE):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    c = np.sin(2 * np.pi * freq * t) * np.exp(-t / (duration * 0.3))
    return c.astype(np.float32)


def add_clicks_to_wav(wav_tensor, beat_np, sr=SAMPLE_RATE, beat_sr=100, subdivisions=4):
    audio = wav_tensor.squeeze(0).numpy().astype(np.float64)
    peak  = np.abs(audio).max() + 1e-8
    spf   = sr // beat_sr
    beat_frames, _ = find_peaks(beat_np, height=0.5, distance=5)
    click_tmpl = make_click(CLICK_FREQ, CLICK_DURATION, sr)
    sub_tmpl   = make_click(SUBCLICK_FREQ, CLICK_DURATION * 0.5, sr)

    def _overlay(buf, pos, tmpl, gain):
        end = min(pos + len(tmpl), len(buf))
        n = end - pos
        if n > 0:
            buf[pos:end] += tmpl[:n] * gain * peak

    for i, bf in enumerate(beat_frames):
        _overlay(audio, int(bf * spf), click_tmpl, CLICK_GAIN)
        if subdivisions > 1 and i + 1 < len(beat_frames):
            gap = beat_frames[i + 1] - bf
            for s in range(1, subdivisions):
                sub_frame = bf + int(gap * s / subdivisions)
                _overlay(audio, int(sub_frame * spf), sub_tmpl, SUBCLICK_GAIN)

    audio = np.clip(audio, -1.0, 1.0)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


class BeatConditionedUNet(nn.Module):
    """CFG-capable beat-conditioned UNet with optional pitch/beat prime support."""
    def __init__(self, pretrained_unet, beat_dim=1):
        super().__init__()
        self.unet = copy.deepcopy(pretrained_unet)
        self.beat_projection = nn.Linear(beat_dim, self.unet.initial_projection.out_channels)
        self.unet.inp_dim = 1

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, x, time, beat):
        x = self.unet.initial_projection(x)
        if beat.ndim == 3: beat = beat.transpose(1, 2)
        elif beat.ndim == 2: beat = beat.unsqueeze(-1)
        beat = self.beat_projection(beat).transpose(1, 2)
        x = x + beat
        time = self.unet.positional_encoding(time)
        def _cat(x_, t_): return torch.cat([x_, t_.unsqueeze(2).expand(-1,-1,x_.shape[-1])], dim=-2)
        skips = []
        for dl in self.unet.downsample_layers:
            skips.append(x); x = _cat(x, time); x = dl(x)
        skips.append(x)
        x = x.permute(0,2,1); x = self.unet.attention_layers(x); x = x.permute(0,2,1)
        for ul in self.unet.upsample_layers:
            x = _cat(x, time); x = torch.cat([x, skips.pop(-1)], dim=1); x = ul(x)
        x = torch.cat([x, skips.pop(-1)], dim=1)
        return self.unet.final_projection(x)

    def sample(self, beat: torch.Tensor, num_steps: int = NUM_STEPS,
               guidance_scale: float = 1.0, prime: torch.Tensor = None):
        b = beat.shape[0]
        noise = torch.randn(b, self.unet.inp_dim, beat.shape[-1]).to(self.device)
        pn,   pad = self.unet.pad_to(noise, self.unet.strides_prod)
        pb,   _   = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        null_beat = torch.zeros_like(pb)
        if prime is not None:
            prime = prime.to(self.device)
        t_arr = torch.ones(b).to(self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                # Prime conditioning: pin first P frames as interpolation
                # between noise and prime, following trajectory
                if prime is not None:
                    alpha_t = (tt * t_arr).unsqueeze(1).unsqueeze(2)
                    P = prime.shape[-1]
                    pn[:, :, :P] = (1 - alpha_t) * noise[:, :, :P] + alpha_t * prime
                # CFG
                pred_cond = self.forward(pn, tt * t_arr, pb)
                if guidance_scale != 1.0:
                    pred_null = self.forward(pn, tt * t_arr, null_beat)
                    pred = pred_null + guidance_scale * (pred_cond - pred_null)
                else:
                    pred = pred_cond
                pn = pn + (1.0 / num_steps) * pred
        return self.unet.unpad(pn, pad)


def qt_to_tokens(pitch_qt_space, qt, min_clip=200):
    tokens = qt.inverse_transform(pitch_qt_space.reshape(-1, 1))
    tokens = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1)
    tokens = torch.round(tokens)
    tokens[tokens < min_clip] = float('nan')
    return tokens


def pitch_to_audio(pitch_npy, qt, audio_model, invert_audio_fn, device, target_len=AUDIO_SEQ_LEN):
    tokens = qt_to_tokens(pitch_npy, qt).to(device)
    interp = p2a.interpolate_pitch(tokens, target_len)
    interp = torch.nan_to_num(interp, nan=196.0).squeeze(1).float().to(device)
    singer_tensor = torch.tensor([SINGER_ID]).to(device)
    noise = torch.randn(1, audio_model.inp_dim, target_len, device=device)
    padded_noise, padding = audio_model.pad_to(noise, audio_model.strides_prod)
    padded_f0, _ = audio_model.pad_to(interp, audio_model.strides_prod)
    t_array = torch.ones(1, device=device)
    strength = 3.0
    with torch.no_grad():
        for t in np.linspace(0, 1, NUM_STEPS + 1)[:-1]:
            tt = torch.tensor(t)
            uncond = audio_model.forward(padded_noise, tt * t_array, padded_f0, singer_tensor,
                                         drop_tokens=False, drop_all=True)
            cond   = audio_model.forward(padded_noise, tt * t_array, padded_f0, singer_tensor,
                                         drop_tokens=False, drop_all=False)
            padded_noise = padded_noise + (1.0 / NUM_STEPS) * (strength * cond + (1 - strength) * uncond)
        audio_samples = audio_model.unpad(padded_noise, padding)
        audio_wav = invert_audio_fn(audio_samples)
    return audio_wav[0].unsqueeze(0).cpu()


def make_beat_prime(beat_gt_np, P):
    """Synthesize a beat prime: silence with short notes at beat positions."""
    synth = np.full(P, SILENCE_QT, dtype=np.float32)
    beat_rises = np.where(np.diff(np.concatenate([[0], (beat_gt_np > 0.5).astype(int)])) == 1)[0]
    for bf in beat_rises[beat_rises < P]:
        synth[bf : bf + NOTE_DUR] = NOTE_QT
    return torch.tensor(synth, dtype=torch.float32).reshape(1, 1, P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",          default="outputs_cfg_prime")
    parser.add_argument("--windows_per_uid",  type=int, default=2)
    parser.add_argument("--skip_existing",    action="store_true", default=True)
    args = parser.parse_args()

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
    print(f"Loaded: {BEST_CKPT}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")

    qt = joblib.load(QT_PATH)

    audio_model, _, _, invert_audio_fn = load_audio_fns(
        audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
    audio_model.eval()
    print("Loaded audio model")

    val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                            val_ratio=0.17, seed=42,
                            window_stride=1200, seq_len=1200)
    all_beats = [val_ds[i]["beat"] for i in range(len(val_ds))]

    # Pick windows_per_uid windows per recording
    seen_uids = {}
    for i in range(len(val_ds)):
        uid = val_ds[i]["uid"]
        if uid not in seen_uids:
            seen_uids[uid] = []
        if len(seen_uids[uid]) < args.windows_per_uid:
            seen_uids[uid].append(i)
    selected = [idx for idxs in seen_uids.values() for idx in idxs]
    print(f"Generating for {len(selected)} windows across {len(seen_uids)} recordings")

    os.makedirs(args.out_dir, exist_ok=True)

    for w_i, idx in enumerate(selected):
        item  = val_ds[idx]
        uid   = item["uid"]
        start = item["start"]
        tag   = f"{uid}_{start}"
        out   = os.path.join(args.out_dir, tag)
        os.makedirs(out, exist_ok=True)

        beat_np = item["beat"].numpy()
        beat_gt = item["beat"].unsqueeze(0).unsqueeze(0).to(device)
        j = (idx + len(val_ds) // 2) % len(val_ds)
        beat_shuf = all_beats[j].unsqueeze(0).unsqueeze(0).to(device)
        beat_zero = torch.zeros_like(beat_gt)

        pitch_gt_np = item["normalized_pitch"].numpy()

        np.save(os.path.join(out, "beat.npy"), beat_np)

        # Build prime tensors
        pitch_prime = torch.tensor(
            pitch_gt_np[:PRIME_LEN], dtype=torch.float32).reshape(1, 1, PRIME_LEN)
        beat_prime = make_beat_prime(beat_np, PRIME_LEN)

        gs = GUIDANCE_SCALE
        gs_tag = f"gs{gs}"

        # All conditions: (beat_tensor, guidance_scale, prime_tensor or None)
        conditions = {
            f"noprime_gt_{gs_tag}":        (beat_gt,   gs,  None),
            f"noprime_shuf_{gs_tag}":      (beat_shuf, gs,  None),
            "noprime_zero":                (beat_zero, 1.0, None),
            f"prime400_gt_{gs_tag}":       (beat_gt,   gs,  pitch_prime),
            f"prime400_shuf_{gs_tag}":     (beat_shuf, gs,  pitch_prime),
            f"beatprime400_gt_{gs_tag}":   (beat_gt,   gs,  beat_prime),
            f"beatprime400_shuf_{gs_tag}": (beat_shuf, gs,  beat_prime),
        }

        beat_frames = np.where(beat_np > 0)[0]
        pitches_qt = {}
        metrics    = {}

        print(f"\n[{w_i+1}/{len(selected)}] {tag}")

        for cond_name, (beat_tensor, scale, prime_tensor) in conditions.items():
            npy_path = os.path.join(out, f"pitch_{cond_name}.npy")
            if args.skip_existing and os.path.exists(npy_path):
                pitches_qt[cond_name] = np.load(npy_path)
                print(f"  {cond_name:35s} EXISTS")
                continue

            gen = beat_model.sample(beat_tensor, guidance_scale=scale,
                                    prime=prime_tensor).squeeze().cpu().numpy()
            np.save(npy_path, gen)
            pitches_qt[cond_name] = gen

            onsets = detect_onsets(gen, qt)
            m = beat_alignment_f1(onsets, beat_frames)
            metrics[cond_name] = m
            print(f"  {cond_name:35s} F1={m['f1']:.3f}  P={m['precision']:.3f}"
                  f"  R={m['recall']:.3f}", flush=True)

        if metrics:
            existing = {}
            metrics_path = os.path.join(out, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    existing = json.load(f)
            existing.update(metrics)
            with open(metrics_path, "w") as f:
                json.dump(existing, f, indent=2)

        # Audio synthesis
        print(f"  Synthesizing audio ...", flush=True)
        for cond_name, pitch_qt_arr in pitches_qt.items():
            wav_path = os.path.join(out, f"audio_{cond_name}.wav")
            if args.skip_existing and os.path.exists(wav_path):
                print(f"  audio {cond_name:30s} EXISTS")
                continue
            wav = pitch_to_audio(pitch_qt_arr, qt, audio_model, invert_audio_fn, device)
            torchaudio.save(wav_path, wav, SAMPLE_RATE)
            wav_click = add_clicks_to_wav(wav, beat_np, subdivisions=4)
            torchaudio.save(os.path.join(out, f"audio_{cond_name}_click.wav"), wav_click, SAMPLE_RATE)
            print(f"  audio {cond_name:30s} saved", flush=True)

        print(f"  Done: {out}")

    print(f"\nAll outputs in {args.out_dir}/")
    print(f"\nTo upload:\n  rclone copy {args.out_dir} gdrive:GaMaDHaNi-samples/{args.out_dir} --progress")


if __name__ == "__main__":
    main()
