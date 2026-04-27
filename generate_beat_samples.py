"""
Generate pitch contours + audio for beat-conditioned GaMaDHaNi.

For a selection of val windows, generates under 3 conditions:
  1. GT beat conditioned
  2. Shuffled beat
  3. Unconditioned (zero beat)

Saves:
  outputs_cfg/
    {uid}_{start}/
      beat.npy                       — GT beat signal
      pitch_gt_gs{scale}.npy         — generated pitch (QT space), GT beat
      pitch_shuf_gs{scale}.npy       — generated pitch, shuffled beat
      pitch_zero.npy                 — generated pitch, zero beat (no guidance)
      audio_gt_gs{scale}.wav
      audio_shuf_gs{scale}.wav
      audio_zero.wav
      plot_gs{scale}.png             — side-by-side pitch + beat overlay
      metrics.json                   — F1 / precision / recall per condition
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
NUM_STEPS   = 100
AUDIO_SEQ_LEN = 750
SINGER_ID   = 3
SAMPLE_RATE = 16000

CLICK_FREQ      = 1000.0   # Hz
CLICK_DURATION  = 0.020    # seconds
CLICK_GAIN      = 0.50     # relative to audio peak
SUBCLICK_GAIN   = 0.20
SUBCLICK_FREQ   = 600.0    # Hz


def make_click(freq, duration, sr=SAMPLE_RATE):
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    c = np.sin(2 * np.pi * freq * t) * np.exp(-t / (duration * 0.3))
    return c.astype(np.float32)


def add_clicks_to_wav(wav_tensor, beat_np, sr=SAMPLE_RATE, beat_sr=100,
                      subdivisions=1):
    """
    Overlay click (and optional subdivision sub-clicks) onto a [1, T] wav tensor.
    subdivisions=1: beats only. subdivisions=2: beats + half-beats, etc.
    """
    audio = wav_tensor.squeeze(0).numpy().astype(np.float64)
    peak  = np.abs(audio).max() + 1e-8
    spf   = sr // beat_sr  # samples per beat frame (160)

    beat_frames, _ = find_peaks(beat_np, height=0.5, distance=5)
    click_tmpl     = make_click(CLICK_FREQ, CLICK_DURATION, sr)
    sub_tmpl       = make_click(SUBCLICK_FREQ, CLICK_DURATION * 0.5, sr)

    def _overlay(buf, pos_samples, template, gain):
        end = min(pos_samples + len(template), len(buf))
        n = end - pos_samples
        if n > 0:
            buf[pos_samples:end] += template[:n] * gain * peak

    for i, bf in enumerate(beat_frames):
        _overlay(audio, int(bf * spf), click_tmpl, CLICK_GAIN)
        if subdivisions > 1 and i + 1 < len(beat_frames):
            gap = beat_frames[i + 1] - bf
            for s in range(1, subdivisions):
                sub_frame = bf + int(gap * s / subdivisions)
                _overlay(audio, int(sub_frame * spf), sub_tmpl, SUBCLICK_GAIN)

    audio = np.clip(audio, -1.0, 1.0)
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


# ── Beat-Conditioned UNet (CFG version — matches train_beat_conditioned.py) ───

class BeatConditionedUNet(nn.Module):
    def __init__(self, pretrained_unet, beat_dim=1, cfg_prob=0.0):
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
               guidance_scale: float = 1.0):
        b = beat.shape[0]
        noise = torch.randn(b, self.unet.inp_dim, beat.shape[-1]).to(self.device)
        pn, pad = self.unet.pad_to(noise, self.unet.strides_prod)
        pb, _   = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        null_beat = torch.zeros_like(pb)
        t_arr = torch.ones(b).to(self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                pred_cond = self.forward(pn, tt * t_arr, pb)
                if guidance_scale != 1.0:
                    pred_null = self.forward(pn, tt * t_arr, null_beat)
                    pred = pred_null + guidance_scale * (pred_cond - pred_null)
                else:
                    pred = pred_cond
                pn = pn + (1.0 / num_steps) * pred
        return self.unet.unpad(pn, pad)


# ── Pitch QT → token space ────────────────────────────────────────────────────

def qt_to_tokens(pitch_qt_space: np.ndarray, qt, min_clip: int = 200) -> torch.Tensor:
    """Convert QT-continuous pitch → integer token space, silence as NaN."""
    tokens = qt.inverse_transform(pitch_qt_space.reshape(-1, 1))
    tokens = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1)
    tokens = torch.round(tokens)
    tokens[tokens < min_clip] = float('nan')
    return tokens


# ── Plot ──────────────────────────────────────────────────────────────────────

def save_plot(uid, start, beat, pitches_hz, labels, path):
    """3-row plot: beat signal + pitch contours (Hz) for each condition."""
    t = np.arange(len(beat)) / 100.0  # seconds
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(4, 1, hspace=0.4)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, beat, color='black', linewidth=0.8)
    ax0.set_title(f"{uid}  start={start}s  — Beat signal")
    ax0.set_ylabel("Beat")
    ax0.set_xlim(t[0], t[-1])

    colors = ['steelblue', 'darkorange', 'forestgreen']
    for i, (pitch_hz, label) in enumerate(zip(pitches_hz, labels)):
        ax = fig.add_subplot(gs[i + 1])
        voiced = pitch_hz > 0
        ax.plot(t[voiced], pitch_hz[voiced], '.', markersize=1, color=colors[i])
        beat_pos = np.where(beat > 0.5)[0]
        for bp in beat_pos:
            ax.axvline(bp / 100.0, color='red', alpha=0.15, linewidth=0.5)
        ax.set_title(label)
        ax.set_ylabel("f0 (Hz)")
        ax.set_xlim(t[0], t[-1])

    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def pitch_to_audio(pitch_npy, qt, audio_model, invert_audio_fn, device, target_len=AUDIO_SEQ_LEN):
    tokens = qt_to_tokens(pitch_npy, qt).to(device)
    interp = p2a.interpolate_pitch(tokens, target_len)
    interp = torch.nan_to_num(interp, nan=196.0).squeeze(1).float().to(device)
    singer_tensor = torch.tensor([SINGER_ID]).to(device)
    # Custom sampling loop — bypasses sample_cfg's hardcoded self.seq_len=750,
    # allowing arbitrary-length audio generation.
    noise = torch.randn(1, audio_model.inp_dim, target_len, device=device)
    padded_noise, padding = audio_model.pad_to(noise, audio_model.strides_prod)
    padded_f0, _          = audio_model.pad_to(interp, audio_model.strides_prod)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default="checkpoints/hmr_gt_beats_cfg/best.ckpt")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--val_ratio",      type=float, default=0.17,
                        help="match training val_ratio (0.10 for hmr_gt_beats, 0.17 for cfg)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--seq_len",        type=int,   default=1200,
                        help="pitch window length in frames (1200=12s, 2000=20s)")
    parser.add_argument("--window_stride",  type=int,   default=1200)
    parser.add_argument("--audio_target_len", type=int, default=None,
                        help="audio mel frames to generate. Default: 750 (12s). "
                             "For 20s use 1250 (= seq_len * 62.5 frames/s / 100 Hz)")
    parser.add_argument("--windows_per_uid", type=int,  default=2,
                        help="how many windows to generate per recording")
    parser.add_argument("--out_dir",        default="outputs_cfg")
    args = parser.parse_args()

    gs_tag = f"gs{args.guidance_scale}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pitch model
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=PITCH_CFG, device=device)

    beat_model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    beat_model.load_state_dict(ckpt["state_dict"])
    beat_model.eval()
    print(f"Loaded: {args.ckpt}  epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.4f}")
    print(f"guidance_scale={args.guidance_scale}  val_ratio={args.val_ratio}")

    qt = joblib.load(QT_PATH)

    # Load audio model
    audio_model, audio_qt, _, invert_audio_fn = load_audio_fns(
        audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
    audio_model.eval()
    print("Loaded audio model")

    # Derive audio target length: default maps pitch frames 1:1 to mel frames
    # Mel frame rate = AUDIO_SEQ_LEN / 12s = 62.5 frames/s; pitch is 100 Hz
    audio_target_len = args.audio_target_len or round(args.seq_len * AUDIO_SEQ_LEN / 1200)
    print(f"Audio target len: {audio_target_len} mel frames "
          f"(≈{audio_target_len/62.5:.1f}s)")

    # Val dataset
    val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                            val_ratio=args.val_ratio, seed=args.seed,
                            window_stride=args.window_stride, seq_len=args.seq_len)
    all_beats = [val_ds[i]["beat"] for i in range(len(val_ds))]

    # Pick N windows per unique UID
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

    for idx in selected:
        item  = val_ds[idx]
        uid   = item["uid"]
        start = item["start"]
        tag   = f"{uid}_{start}"
        out   = os.path.join(args.out_dir, tag)
        os.makedirs(out, exist_ok=True)

        beat_np   = item["beat"].numpy()
        beat_gt   = item["beat"].unsqueeze(0).unsqueeze(0).to(device)
        j         = (idx + len(val_ds) // 2) % len(val_ds)
        beat_shuf = all_beats[j].unsqueeze(0).unsqueeze(0).to(device)
        beat_zero = torch.zeros_like(beat_gt)
        beat_frames = np.where(beat_np > 0)[0]

        np.save(os.path.join(out, "beat.npy"), beat_np)

        # GT and shuffled use guidance_scale; zero-beat unconditioned uses scale=1.0
        conditions = {
            f"gt_{gs_tag}":   (beat_gt,   args.guidance_scale),
            f"shuf_{gs_tag}": (beat_shuf, args.guidance_scale),
            "zero":           (beat_zero, 1.0),
        }

        pitches_qt = {}
        pitches_hz = {}
        metrics    = {}

        print(f"  [{tag}] generating pitch ...", flush=True)
        for cond_name, (beat_tensor, scale) in conditions.items():
            gen = beat_model.sample(beat_tensor, guidance_scale=scale).squeeze().cpu().numpy()
            pitches_qt[cond_name] = gen
            pitches_hz[cond_name] = invert_pitch_fn(f0=gen).flatten()
            np.save(os.path.join(out, f"pitch_{cond_name}.npy"), gen)

            onsets = detect_onsets(gen, qt)
            m = beat_alignment_f1(onsets, beat_frames)
            metrics[cond_name] = m
            print(f"    {cond_name:20s}  F1={m['f1']:.3f}  P={m['precision']:.3f}"
                  f"  R={m['recall']:.3f}  onsets={m['n_onsets']}", flush=True)

        with open(os.path.join(out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Plot
        gt_key   = f"gt_{gs_tag}"
        shuf_key = f"shuf_{gs_tag}"
        save_plot(uid, start, beat_np,
                  [pitches_hz[gt_key], pitches_hz[shuf_key], pitches_hz["zero"]],
                  [f"GT beat (gs={args.guidance_scale})",
                   f"Shuffled beat (gs={args.guidance_scale})",
                   "Unconditioned (zero beat)"],
                  os.path.join(out, f"plot_{gs_tag}.png"))

        # Convert to audio + add click tracks
        print(f"  [{tag}] converting to audio ...", flush=True)
        for cond_name in conditions:
            wav = pitch_to_audio(pitches_qt[cond_name], qt, audio_model, invert_audio_fn, device,
                                 target_len=audio_target_len)
            wav_path = os.path.join(out, f"audio_{cond_name}.wav")
            torchaudio.save(wav_path, wav, SAMPLE_RATE)
            # click track (beats + subdivisions)
            wav_click = add_clicks_to_wav(wav, beat_np, subdivisions=4)
            torchaudio.save(os.path.join(out, f"audio_{cond_name}_click.wav"), wav_click, SAMPLE_RATE)
            print(f"    Saved {wav_path}", flush=True)

        print(f"  Done: {out}")

    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
