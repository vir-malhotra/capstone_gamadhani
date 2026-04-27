"""
Beat alignment evaluation for beat-conditioned GaMaDHaNi.

For each val recording, generates pitch contours under 3 conditions:
  1. GT beat conditioned   — our fine-tuned model + correct beat signal
  2. Shuffled beat         — our fine-tuned model + wrong beat signal (sanity check)
  3. Unconditioned         — original GaMaDHaNi, no beat input

Then computes beat alignment F1: fraction of note onsets that fall near beat positions.

If conditioning is working: GT > shuffled ≈ unconditioned.
"""

import copy, os, sys, random
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG     = "configs/diffusion_pitch_config.gin"
NUM_STEPS      = 100
NUM_SAMPLES    = 4    # generations per recording
ONSET_WINDOW   = 5    # ±5 frames (±50ms) to count a beat as aligned
ONSET_JUMP_THR = 10   # token-space jump threshold to count as onset (10 tokens ≈ 1 semitone)


# ── Model ────────────────────────────────────────────────────────────────────

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
               prime: torch.Tensor = None, guidance_scale: float = 1.0):
        b = beat.shape[0]
        noise = torch.randn(b, self.unet.inp_dim, beat.shape[-1]).to(self.device)
        pn, pad = self.unet.pad_to(noise, self.unet.strides_prod)
        pb, _   = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        null_beat = torch.zeros_like(pb)
        if prime is not None:
            prime = prime.to(self.device)
        t_arr = torch.ones(b).to(self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                alpha_t = (tt * t_arr).unsqueeze(1).unsqueeze(2)
                if prime is not None:
                    P = prime.shape[-1]
                    pn[:, :, :P] = (1 - alpha_t) * noise[:, :, :P] + alpha_t * prime
                pred_cond = self.forward(pn, tt * t_arr, pb)
                if guidance_scale != 1.0:
                    pred_null = self.forward(pn, tt * t_arr, null_beat)
                    pred = pred_null + guidance_scale * (pred_cond - pred_null)
                else:
                    pred = pred_cond
                pn = pn + (1.0 / num_steps) * pred
        return self.unet.unpad(pn, pad)


# ── Onset detection ──────────────────────────────────────────────────────────

def detect_onsets(qt_output: np.ndarray, qt,
                  jump_thr: int = ONSET_JUMP_THR) -> np.ndarray:
    """
    Detect note onsets from model output (QT-continuous space).

    Two onset types:
      1. Silence → voiced: token jumps from <200 (silence region) to ≥200.
      2. Large pitch jump: |token[t] - token[t-1]| >= jump_thr (≈ 1 semitone).

    The QT inverse maps the continuous model output back to the token space
    used during training (roughly [196, 600] for silence+voiced range).
    """
    tokens = qt.inverse_transform(qt_output.reshape(-1, 1)).flatten()

    silence = tokens < 200
    # type-1: silence→voiced
    sil_to_voiced = np.where(silence[:-1] & ~silence[1:])[0] + 1

    # type-2: large pitch jump (among voiced frames)
    jumps = np.abs(np.diff(tokens))
    big_jump = (jumps >= jump_thr) & ~silence[1:]
    jump_onsets = np.where(big_jump)[0] + 1

    onsets = np.unique(np.concatenate([sil_to_voiced, jump_onsets]))
    return onsets


# ── Beat alignment F1 ────────────────────────────────────────────────────────

def beat_alignment_f1(onsets: np.ndarray, beat_frames: np.ndarray,
                      window: int = ONSET_WINDOW) -> dict:
    """
    Compute F1 between note onsets and beat positions.
    A beat is a TP if there's an onset within ±window frames.
    An onset is a FP if it's not near any beat.
    """
    if len(onsets) == 0 or len(beat_frames) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_onsets": 0, "n_beats": len(beat_frames)}

    tp = 0
    matched_onsets = set()
    for b in beat_frames:
        near = np.where(np.abs(onsets - b) <= window)[0]
        if len(near) > 0:
            tp += 1
            matched_onsets.add(near[0])

    fp = len(onsets) - len(matched_onsets)
    fn = len(beat_frames) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"f1": f1, "precision": precision, "recall": recall,
            "n_onsets": len(onsets), "n_beats": len(beat_frames)}


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    import json
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)

    qt = joblib.load(QT_PATH)

    beat_model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    beat_model.load_state_dict(ckpt["state_dict"])
    beat_model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    print(f"  epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")
    print(f"  guidance_scale={args.guidance_scale}, val_ratio={args.val_ratio}")

    # Val dataset — use matching val_ratio/seed for the checkpoint
    val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                            val_ratio=args.val_ratio, seed=args.seed,
                            window_stride=args.window_stride, seq_len=args.seq_len)
    print(f"Val windows: {len(val_ds)}")

    # Collect all beat signals for shuffling
    all_beats = [val_ds[i]["beat"] for i in range(len(val_ds))]

    results = {"gt_conditioned": [], "shuffled_beat": [], "unconditioned": []}

    for i in range(len(val_ds)):
        item = val_ds[i]
        beat_gt = item["beat"].unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]
        beat_frames = np.where(item["beat"].numpy() > 0)[0]

        # pick a different recording's beat for the shuffle condition
        j = (i + len(val_ds) // 2) % len(val_ds)
        beat_shuf = all_beats[j].unsqueeze(0).unsqueeze(0).to(device)

        # zero beat for unconditioned proxy
        beat_zero = torch.zeros_like(beat_gt)

        for _ in range(args.num_samples):
            gs = args.guidance_scale

            # GT conditioned
            gen_gt   = beat_model.sample(beat_gt, guidance_scale=gs).squeeze().cpu().numpy()
            onsets   = detect_onsets(gen_gt, qt)
            results["gt_conditioned"].append(beat_alignment_f1(onsets, beat_frames))

            # Shuffled beat
            gen_shuf = beat_model.sample(beat_shuf, guidance_scale=gs).squeeze().cpu().numpy()
            onsets   = detect_onsets(gen_shuf, qt)
            results["shuffled_beat"].append(beat_alignment_f1(onsets, beat_frames))

            # Unconditioned (zero beat — model gets no useful signal)
            gen_zero = beat_model.sample(beat_zero, guidance_scale=1.0).squeeze().cpu().numpy()
            onsets   = detect_onsets(gen_zero, qt)
            results["unconditioned"].append(beat_alignment_f1(onsets, beat_frames))

        print(f"  [{i+1}/{len(val_ds)}] uid={item['uid']}  beats={len(beat_frames)}", flush=True)

    # ── Summary ──
    print("\n" + "="*60)
    print(f"ckpt={args.ckpt}  guidance_scale={args.guidance_scale}")
    print(f"{'Condition':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Onsets':>7}")
    print("-"*60)
    for cond, rows in results.items():
        f1s  = [r["f1"]        for r in rows]
        prec = [r["precision"] for r in rows]
        rec  = [r["recall"]    for r in rows]
        nos  = [r["n_onsets"]  for r in rows]
        print(f"{cond:<22} {np.mean(f1s):>6.3f} {np.mean(prec):>6.3f} "
              f"{np.mean(rec):>6.3f} {np.mean(nos):>7.1f}")
    print("="*60)
    gt_f1  = np.mean([r["f1"] for r in results["gt_conditioned"]])
    shf_f1 = np.mean([r["f1"] for r in results["shuffled_beat"]])
    unc_f1 = np.mean([r["f1"] for r in results["unconditioned"]])
    print(f"GT−shuffled gap: {gt_f1 - shf_f1:+.3f}   GT−uncond gap: {gt_f1 - unc_f1:+.3f}")

    out = {"args": vars(args), "results": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",           default="checkpoints/hmr_gt_beats_cfg/best.ckpt")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--val_ratio",      type=float, default=0.17,
                        help="must match training val_ratio (0.10 for hmr_gt_beats, 0.17 for cfg/val19)")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--num_samples",    type=int,   default=NUM_SAMPLES)
    parser.add_argument("--seq_len",        type=int,   default=1200)
    parser.add_argument("--window_stride",  type=int,   default=1200)
    parser.add_argument("--output",         default="beat_alignment_results.json")
    args = parser.parse_args()
    evaluate(args)
