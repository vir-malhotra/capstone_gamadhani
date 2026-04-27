"""
Test whether Stage 2 (UNetPitchConditioned) can generate audio longer than
the training length (AUDIO_SEQ_LEN=750 frames ≈ 7.5s equivalent).

Strategy: bypass sample_cfg()'s hardcoded noise=self.seq_len and instead
create noise at a custom target_len, then run the CFG sampling loop manually
using forward() directly.

Usage:
    python test_long_audio.py --target_len 1500   # 2x normal
    python test_long_audio.py --target_len 750    # baseline (sanity check)
"""

import argparse, copy, os, sys
import numpy as np
import torch
import torchaudio
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns, load_audio_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset
import gamadhani.utils.pitch_to_audio_utils as p2a

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
AUDIO_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-model.ckpt"
AUDIO_QT   = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/pitch_to_audio_model-qt.joblib"
PITCH_CFG  = "configs/diffusion_pitch_config.gin"
AUDIO_CFG  = "configs/pitch_to_audio_config.gin"
CKPT       = "checkpoints/hmr_gt_beats_cfg/best.ckpt"
NUM_STEPS  = 100
SINGER_ID  = 3
SAMPLE_RATE = 16000
AUDIO_SEQ_LEN = 750  # training length (reference)


import torch.nn as nn

class BeatConditionedUNet(nn.Module):
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

    def sample(self, beat, num_steps=NUM_STEPS, guidance_scale=3.0):
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
                pred_null = self.forward(pn, tt * t_arr, null_beat)
                pred = pred_null + guidance_scale * (pred_cond - pred_null)
                pn = pn + (1.0 / num_steps) * pred
        return self.unet.unpad(pn, pad)


def sample_audio_at_length(audio_model, f0_tokens, target_len, singer_id,
                            num_steps=NUM_STEPS, strength=3.0, device="cuda"):
    """
    Run Stage 2 sampling with noise initialized at target_len instead of
    the hardcoded self.seq_len=750. f0_tokens are interpolated to target_len.
    """
    # Interpolate pitch to target_len
    f0_interp = p2a.interpolate_pitch(f0_tokens, target_len)
    f0_interp = torch.nan_to_num(f0_interp, nan=196.0).squeeze(1).float().to(device)
    print(f"  f0 interpolated: {f0_interp.shape}")

    # Create noise at target_len (instead of self.seq_len=750)
    noise = torch.randn(1, audio_model.inp_dim, target_len, device=device)
    print(f"  noise shape: {noise.shape}")

    padded_noise, padding = audio_model.pad_to(noise, audio_model.strides_prod)
    padded_f0, _          = audio_model.pad_to(f0_interp, audio_model.strides_prod)
    print(f"  padded noise: {padded_noise.shape}  padded f0: {padded_f0.shape}")

    singer = torch.tensor([singer_id], device=device)
    t_array = torch.ones(1, device=device)

    with torch.no_grad():
        for t in np.linspace(0, 1, num_steps + 1)[:-1]:
            tt = torch.tensor(t)
            uncond = audio_model.forward(padded_noise, tt * t_array, padded_f0, singer,
                                         drop_tokens=False, drop_all=True)
            cond   = audio_model.forward(padded_noise, tt * t_array, padded_f0, singer,
                                         drop_tokens=False, drop_all=False)
            total  = strength * cond + (1 - strength) * uncond
            padded_noise = padded_noise + (1.0 / num_steps) * total

    return audio_model.unpad(padded_noise, padding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT,
                        help="beat model checkpoint")
    parser.add_argument("--target_len", type=int, default=1500,
                        help="audio sequence length to generate (750=baseline, 1500=2x)")
    parser.add_argument("--seq_len", type=int, default=1200,
                        help="pitch window length (1200=12s, 2000=20s)")
    parser.add_argument("--window_stride", type=int, default=1200)
    parser.add_argument("--val_ratio", type=float, default=0.17)
    parser.add_argument("--window_idx", type=int, default=0,
                        help="which val window to use")
    parser.add_argument("--out", default="test_long_audio_out")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Load pitch model
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=PITCH_CFG, device=device)
    qt = joblib.load(QT_PATH)

    beat_model = BeatConditionedUNet(pitch_model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    beat_model.load_state_dict(ckpt["state_dict"])
    beat_model.eval()
    print(f"Loaded pitch model: epoch={ckpt['epoch']} val_loss={ckpt['val_loss']:.4f}")

    # Load audio model
    audio_model, audio_qt, _, invert_audio_fn = load_audio_fns(
        audio_path=AUDIO_PATH, qt_path=AUDIO_QT, config_path=AUDIO_CFG, device=device)
    audio_model.eval()
    print(f"Audio model: inp_dim={audio_model.inp_dim}  seq_len={audio_model.seq_len}  strides_prod={audio_model.strides_prod}")

    # Val dataset
    val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                            val_ratio=args.val_ratio, seed=42,
                            window_stride=args.window_stride, seq_len=args.seq_len)
    idx = args.window_idx % len(val_ds)
    item = val_ds[idx]
    print(f"Window: uid={item['uid']}  start={item['start']}  beat shape={item['beat'].shape}")

    beat_gt = item["beat"].unsqueeze(0).unsqueeze(0).to(device)

    # Generate pitch
    print(f"\nGenerating pitch ({args.seq_len} frames) ...")
    with torch.no_grad():
        pitch_qt_space = beat_model.sample(beat_gt, guidance_scale=3.0).squeeze().cpu().numpy()
    print(f"  pitch shape: {pitch_qt_space.shape}")

    tokens = qt.inverse_transform(pitch_qt_space.reshape(-1, 1))
    tokens_t = torch.tensor(tokens, dtype=torch.float32).reshape(1, 1, -1)
    tokens_t = torch.round(tokens_t)
    tokens_t[tokens_t < 200] = float('nan')

    # Generate audio at baseline (750) for comparison
    print(f"\nGenerating audio at baseline len={AUDIO_SEQ_LEN} ...")
    audio_baseline = sample_audio_at_length(
        audio_model, tokens_t, AUDIO_SEQ_LEN, SINGER_ID, device=device)
    wav_baseline = invert_audio_fn(audio_baseline)[0].unsqueeze(0).cpu()
    path_b = os.path.join(args.out, f"audio_baseline_{AUDIO_SEQ_LEN}.wav")
    torchaudio.save(path_b, wav_baseline, SAMPLE_RATE)
    print(f"  Saved {path_b}  shape={wav_baseline.shape}  duration={wav_baseline.shape[-1]/SAMPLE_RATE:.1f}s")

    # Generate audio at target_len
    print(f"\nGenerating audio at target len={args.target_len} ...")
    audio_long = sample_audio_at_length(
        audio_model, tokens_t, args.target_len, SINGER_ID, device=device)
    wav_long = invert_audio_fn(audio_long)[0].unsqueeze(0).cpu()
    path_l = os.path.join(args.out, f"audio_long_{args.target_len}.wav")
    torchaudio.save(path_l, wav_long, SAMPLE_RATE)
    print(f"  Saved {path_l}  shape={wav_long.shape}  duration={wav_long.shape[-1]/SAMPLE_RATE:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
