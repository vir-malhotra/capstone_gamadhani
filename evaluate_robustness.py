"""
Robustness evaluation: compare original vs augmented model under degraded beat signals.

For each val window, both models are run with:
  1. GT beat (clean baseline)
  2. Section dropout at 10%, 20%, 30%, 50% of the window
  3. Random pulse dropout at p=0.1, 0.2, 0.3, 0.5
  4. Beat jitter ±2, ±5, ±10, ±20 frames

Results saved to robustness_results.json and printed as a comparison table.
"""

import copy, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset, _extract_events, _render_beat_pulses

PITCH_PATH  = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH     = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG      = "configs/diffusion_pitch_config.gin"
ORIG_CKPT   = "checkpoints/hmr_gt_beats/best.ckpt"
AUG_CKPT    = "checkpoints/hmr_gt_beats_augmented/best.ckpt"
NUM_STEPS   = 100
NUM_SAMPLES = 2   # fewer samples per window to keep runtime reasonable
ONSET_THR   = 10
WINDOW      = 5


# ── BeatConditionedUNet ───────────────────────────────────────────────────────

class BeatConditionedUNet(nn.Module):
    def __init__(self, pretrained_unet, beat_dim=1, beat_dropout=0.1):
        super().__init__()
        self.unet = copy.deepcopy(pretrained_unet)
        self.beat_projection = nn.Linear(beat_dim, self.unet.initial_projection.out_channels)
        self.beat_dropout = nn.Dropout(beat_dropout)
        self.unet.inp_dim = 1

    @property
    def device(self): return next(self.parameters()).device

    def forward(self, x, time, beat, drop=True):
        x = self.unet.initial_projection(x)
        if beat.ndim == 3: beat = beat.transpose(1, 2)
        elif beat.ndim == 2: beat = beat.unsqueeze(-1)
        beat = self.beat_projection(beat).transpose(1, 2)
        if drop: beat = self.beat_dropout(beat)
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

    def sample(self, beat, num_steps=NUM_STEPS):
        b = beat.shape[0]
        noise = torch.randn(b, self.unet.inp_dim, self.unet.seq_len).to(self.device)
        pn, pad = self.unet.pad_to(noise, self.unet.strides_prod)
        pb, _   = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        t_arr   = torch.ones(b).to(self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                pn = pn + (1.0 / num_steps) * self.forward(pn, tt * t_arr, pb, drop=False)
        return self.unet.unpad(pn, pad)


# ── Beat degradation functions ────────────────────────────────────────────────

def degrade_section_dropout(beat, frac, rng):
    """Zero out a contiguous block of length frac*T, placed randomly."""
    b = beat.copy()
    T = len(b)
    length = max(1, int(T * frac))
    start  = int(rng.integers(0, T - length))
    b[start:start + length] = 0.0
    return b

def degrade_pulse_dropout(beat, p, rng):
    """Remove each beat event independently with probability p."""
    events = _extract_events(beat)
    if len(events) == 0: return beat.copy()
    keep   = rng.random(len(events)) >= p
    events = events[keep]
    return _render_beat_pulses(events, len(beat))

def degrade_jitter(beat, max_shift, rng):
    """Shift each beat event by a random offset in [-max_shift, +max_shift]."""
    events = _extract_events(beat)
    if len(events) == 0: return beat.copy()
    shifts = rng.integers(-max_shift, max_shift + 1, size=len(events))
    events = np.clip(events + shifts, 0, len(beat) - 1)
    return _render_beat_pulses(events, len(beat))


# ── Metrics ───────────────────────────────────────────────────────────────────

def get_onsets(pqt, qt):
    tokens = qt.inverse_transform(pqt.reshape(-1,1)).flatten()
    return np.where(np.abs(np.diff(tokens)) >= ONSET_THR)[0] + 1

def compute_f1(onsets, beats, win=WINDOW):
    if len(onsets) == 0 or len(beats) == 0:
        return 0.0
    tp, matched = 0, set()
    for b in beats:
        near = np.where(np.abs(onsets - b) <= win)[0]
        if len(near): tp += 1; matched.add(near[0])
    fp = len(onsets) - len(matched)
    fn = len(beats) - tp
    p  = tp/(tp+fp) if tp+fp else 0.
    r  = tp/(tp+fn) if tp+fn else 0.
    return 2*p*r/(p+r) if p+r else 0.

def beat_events_for_f1(beat):
    """Get GT beat event frame indices for F1 evaluation (always uses original GT)."""
    return _extract_events(beat)


# ── Main ──────────────────────────────────────────────────────────────────────

def load_model(ckpt_path, pretrained_unet, device):
    model = BeatConditionedUNet(pretrained_unet).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    ep = ckpt["epoch"]; vl = ckpt["val_loss"]
    print(f"  Loaded {ckpt_path}: epoch={ep}, val_loss={vl:.4f}")
    return model


def run_condition(model, beat_signal, qt, device, n_samples=NUM_SAMPLES):
    """Generate n_samples and return mean F1 against the GT beat events."""
    beat_gt_events = beat_events_for_f1(beat_signal)
    if len(beat_gt_events) == 0:
        return 0.0
    beat_t = torch.tensor(beat_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    f1s = []
    for _ in range(n_samples):
        gen    = model.sample(beat_t).squeeze().cpu().numpy()
        onsets = get_onsets(gen, qt)
        f1s.append(compute_f1(onsets, beat_gt_events))
    return float(np.mean(f1s))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng    = np.random.default_rng(42)

    print("Loading pitch model...")
    pitch_model, pitch_qt, pitch_task_fn, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)
    qt = joblib.load(QT_PATH)

    print("Loading original model...")
    orig_model = load_model(ORIG_CKPT, pitch_model, device)
    print("Loading augmented model...")
    aug_model  = load_model(AUG_CKPT,  pitch_model, device)

    val_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val", val_ratio=0.1, seed=42,
                             beat_augment=False)
    print(f"Val windows: {len(val_ds)}\n")

    # Define degradation conditions
    conditions = [("gt_clean", None)]
    for frac in [0.1, 0.2, 0.3, 0.5]:
        conditions.append((f"section_{int(frac*100)}pct", ("section", frac)))
    for p in [0.1, 0.2, 0.3, 0.5]:
        conditions.append((f"pulse_drop_p{int(p*10)}", ("pulse", p)))
    for shift in [2, 5, 10, 20]:
        conditions.append((f"jitter_{shift}fr", ("jitter", shift)))

    results = {cname: {"orig": [], "aug": []} for cname, _ in conditions}

    for idx in range(len(val_ds)):
        item  = val_ds[idx]
        beat  = item["beat"].numpy()
        uid   = item["uid"]
        print(f"  [{idx+1}/{len(val_ds)}] uid={uid}", flush=True)

        for cname, deg in conditions:
            if deg is None:
                deg_beat = beat
            elif deg[0] == "section":
                deg_beat = degrade_section_dropout(beat, deg[1], rng)
            elif deg[0] == "pulse":
                deg_beat = degrade_pulse_dropout(beat, deg[1], rng)
            else:  # jitter
                deg_beat = degrade_jitter(beat, deg[1], rng)

            f1_orig = run_condition(orig_model, deg_beat, qt, device)
            f1_aug  = run_condition(aug_model,  deg_beat, qt, device)
            results[cname]["orig"].append(f1_orig)
            results[cname]["aug"].append(f1_aug)

    # Print summary table
    print("\n" + "="*65)
    print(f"{'Condition':<22}  {'Orig F1':>8}  {'Aug F1':>8}  {'Delta':>8}")
    print("-"*65)
    for cname, _ in conditions:
        o = np.mean(results[cname]["orig"])
        a = np.mean(results[cname]["aug"])
        print(f"{cname:<22}  {o:>8.3f}  {a:>8.3f}  {a-o:>+8.3f}")
    print("="*65)

    with open("robustness_results.json", "w") as f:
        json.dump({k: {m: [float(x) for x in v] for m, v in vd.items()}
                   for k, vd in results.items()}, f, indent=2)
    print("Saved robustness_results.json")


if __name__ == "__main__":
    main()
