"""
Phase distribution eval for beat-conditioned GaMaDHaNi.

For each generated onset, computes its phase within the beat cycle:
  phase = 0 → onset is exactly on a beat
  phase = 0.5 → onset is halfway between beats

If conditioning works:
  - GT beat condition → distribution spiked near phase 0
  - Shuffled / zero → uniform distribution

Also:
  - Bootstrap CI on GT−shuf cross-correlation gap
  - Laya stratification of all metrics
"""

import os, sys, json, copy, random
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from scipy.stats import bootstrap as scipy_bootstrap
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns
import lmdb
from gamadhani.src.protobuf.data_example import AudioExample

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG     = "configs/diffusion_pitch_config.gin"
GT_CKPT    = "checkpoints/hmr_gt_beats/best.ckpt"
EXT_CKPT   = "checkpoints/hmr_extracted_beats/best.ckpt"
GT_LMDB    = "/home/vm2426/HMR_processed/lmdb/val"
EXT_LMDB   = "/home/vm2426/HMR_processed/lmdb_extracted/val"
INSTR_CSV  = "/home/vm2426/beat_transformer_hindustani/inference/hmr_instruments.csv"

NUM_STEPS   = 50
NUM_SAMPLES = 4
SEQ_LEN     = 1200
ONSET_THR   = 10
MAX_LAG     = 50
FPS         = 100
N_BOOT      = 2000
N_PHASE_BINS = 20
EXCLUDE_LAYAS = ["vilambit"]


# ── Model ─────────────────────────────────────────────────────────────────────

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def detect_onsets(norm_pitch, qt, thr=ONSET_THR):
    tokens = qt.inverse_transform(norm_pitch.reshape(-1, 1)).flatten()
    silence = tokens < 200
    sil_to_v = np.where(silence[:-1] & ~silence[1:])[0] + 1
    jumps = np.abs(np.diff(tokens))
    big   = (jumps >= thr) & ~silence[1:]
    return np.unique(np.concatenate([sil_to_v, np.where(big)[0] + 1]))


def onset_phases(onsets, beat_frames):
    """
    For each onset, compute its fractional phase within the beat cycle.
    phase=0 means onset is exactly on a beat; phase=0.5 means midway between.
    Returns array of phases in [0, 1).
    """
    if len(beat_frames) < 2 or len(onsets) == 0:
        return np.array([])
    phases = []
    for o in onsets:
        # find surrounding beats
        before = beat_frames[beat_frames <= o]
        after  = beat_frames[beat_frames >  o]
        if len(before) == 0 or len(after) == 0:
            continue
        b0, b1 = before[-1], after[0]
        period = b1 - b0
        if period <= 0:
            continue
        phases.append((o - b0) / period)
    return np.array(phases)


def peak_crosscorr(onset_sig, beat_sig, max_lag=MAX_LAG):
    if onset_sig.sum() == 0 or beat_sig.sum() == 0:
        return 0.0, 0
    o = onset_sig / (onset_sig.std() + 1e-8)
    b = beat_sig  / (beat_sig.std()  + 1e-8)
    corr = np.correlate(o, b, mode='full')
    corr /= len(onset_sig)
    center = len(corr) // 2
    window = corr[center - max_lag : center + max_lag + 1]
    peak_idx = np.argmax(window)
    return float(window[peak_idx]), peak_idx - max_lag


def onset_signal(onsets, length):
    sig = np.zeros(length, dtype=np.float32)
    sig[onsets[onsets < length]] = 1.0
    return sig


def beat_signal_from_frames(beat_frames, length):
    sig = np.zeros(length, dtype=np.float32)
    sig[beat_frames[beat_frames < length]] = 1.0
    return sig


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eval(model, windows, gt_beat_windows, qt, laya_map, device, label):
    """
    windows:         list of dicts with uid/start/beat — used as conditioning input
    gt_beat_windows: dict of (uid, start) -> gt_beat_array — used for phase/CC evaluation
    """
    records = defaultdict(list)

    for i, w in enumerate(windows):
        uid  = w["uid"]
        laya = laya_map.get(uid, "unknown")

        if laya in EXCLUDE_LAYAS:
            continue

        # GT beat frames for evaluation (always from GT annotations)
        gt_beat_arr = gt_beat_windows.get((uid, w["start"]))
        if gt_beat_arr is None:
            continue
        gt_wins = np.where(np.diff(np.concatenate([[0], (gt_beat_arr > 0.5).astype(int)])) == 1)[0]
        if len(gt_wins) < 3:
            continue

        beat_in   = torch.tensor(w["beat"]).unsqueeze(0).unsqueeze(0).to(device)
        shuf_cands = [ww for ww in windows if ww["uid"] != uid]
        shuf_beat  = torch.tensor(random.choice(shuf_cands)["beat"]).unsqueeze(0).unsqueeze(0).to(device)
        beat_zero  = torch.zeros_like(beat_in)

        beat_sig = beat_signal_from_frames(gt_wins, SEQ_LEN)

        for _ in range(NUM_SAMPLES):
            for cond_name, cond_beat in [("gt", beat_in), ("shuf", shuf_beat), ("zero", beat_zero)]:
                gen    = model.sample(cond_beat).squeeze().cpu().numpy()
                onsets = detect_onsets(gen, qt)
                o_sig  = onset_signal(onsets, SEQ_LEN)

                phases  = onset_phases(onsets, gt_wins)
                cc, lag = peak_crosscorr(o_sig, beat_sig)

                records[cond_name].append({
                    "phases": phases.tolist(),
                    "cc": cc,
                    "lag": int(lag),
                    "laya": laya,
                    "n_onsets": int(len(onsets)),
                    "n_beats": int(len(gt_wins)),
                })

        print(f"  [{i+1}/{len(windows)}]", end="\r", flush=True)

    print(f"\n\nDone. Summarizing {label}...")
    print_results(records, label)
    return records


def print_results(records, label):
    def phase_concentration(phases_list):
        all_phases = np.concatenate([np.array(p) for p in phases_list if len(p) > 0])
        if len(all_phases) == 0:
            return np.nan, np.nan, 0
        angles = 2 * np.pi * all_phases
        R = np.abs(np.mean(np.exp(1j * angles)))
        near_beat = np.mean((all_phases < 0.1) | (all_phases > 0.9))
        return float(R), float(near_beat), len(all_phases)

    def mean_fn(x, axis): return np.mean(x, axis=axis)

    print(f"\n{'='*70}")
    print(f"PHASE DISTRIBUTION — {label}")
    print(f"{'='*70}")
    print(f"{'Condition':<10} {'Rayleigh R':>12} {'Near-beat%':>12} {'N phases':>10}")
    print(f"{'-'*50}")
    for cond in ["gt", "shuf", "zero"]:
        phases_list = [r["phases"] for r in records[cond]]
        R, nb, n = phase_concentration(phases_list)
        print(f"  {cond:<8}   {R:>11.4f}  {nb:>11.3f}  {n:>9d}")

    gt_cc   = np.array([r["cc"] for r in records["gt"]])
    shuf_cc = np.array([r["cc"] for r in records["shuf"]])
    gaps    = gt_cc - shuf_cc
    boot = scipy_bootstrap((gaps,), mean_fn, n_resamples=N_BOOT,
                           confidence_level=0.95, random_state=42)
    ci_lo, ci_hi = boot.confidence_interval
    print(f"\n  Bootstrap CI on GT−shuf CC gap:")
    print(f"  Observed: {np.mean(gaps):+.4f}  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]  "
          f"Significant: {'YES' if ci_lo > 0 else 'NO'}")

    print(f"\n{'Laya':<16} {'N':>5} {'CC gap':>8} {'CI lo':>8} {'CI hi':>8} {'R(gt)':>7} {'R(shuf)':>8} {'NB%(gt)':>8}")
    print(f"{'-'*72}")
    layas = sorted(set(r["laya"] for r in records["gt"]))
    for laya in layas:
        gt_sub   = [r for r in records["gt"]   if r["laya"] == laya]
        shuf_sub = [r for r in records["shuf"] if r["laya"] == laya]
        if len(gt_sub) < 10:
            continue
        gaps_l = np.array([r["cc"] for r in gt_sub]) - np.array([r["cc"] for r in shuf_sub])
        try:
            b = scipy_bootstrap((gaps_l,), mean_fn, n_resamples=N_BOOT,
                                confidence_level=0.95, random_state=42)
            lo, hi = b.confidence_interval
        except Exception:
            lo, hi = float("nan"), float("nan")
        R_gt, nb_gt, _ = phase_concentration([r["phases"] for r in gt_sub])
        R_shuf, _, _   = phase_concentration([r["phases"] for r in shuf_sub])
        print(f"  {laya:<14}  {len(gt_sub):>5}  {np.mean(gaps_l):>+8.4f}  {lo:>+8.4f}  {hi:>+8.4f}  "
              f"{R_gt:>7.4f}  {R_shuf:>8.4f}  {nb_gt:>8.3f}")

    # Phase histogram
    print(f"\n  Phase histograms (GT beats model = {label}):")
    bin_edges = np.linspace(0, 1, N_PHASE_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for cond in ["gt", "shuf", "zero"]:
        all_phases = np.concatenate([np.array(r["phases"]) for r in records[cond] if len(r["phases"]) > 0])
        hist, _ = np.histogram(all_phases, bins=bin_edges)
        hist = hist / hist.sum()
        uniform = 1.0 / N_PHASE_BINS
        print(f"\n  {cond.upper()} (n={len(all_phases)}):")
        for c, h in zip(bin_centers, hist):
            bar = "█" * int(h / uniform * 10)
            marker = " ← beat" if c < (1.0 / N_PHASE_BINS) else ""
            print(f"    {c:.2f} | {bar:<20} {h:.3f}{marker}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qt = joblib.load(QT_PATH)

    df = pd.read_csv(INSTR_CSV)
    df['UID'] = df['UID'].astype(str)
    laya_map = dict(zip(df['UID'], df['Laya'].str.lower().str.strip()))

    pitch_model, _, _, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)

    def load_beat_windows(lmdb_path):
        windows = {}
        env = lmdb.open(lmdb_path, lock=False, readahead=False)
        with env.begin() as txn:
            for key in txn.cursor().iternext(values=False):
                uid = key.decode("ascii")
                ae  = AudioExample(txn.get(key))
                beat_full = ae.as_dict()["beat"]["data"].astype(np.float32)[0]
                n = beat_full.shape[0]
                for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                    windows[(uid, start)] = beat_full[start:start+SEQ_LEN]
        env.close()
        return windows

    gt_beat_windows  = load_beat_windows(GT_LMDB)
    ext_beat_windows = load_beat_windows(EXT_LMDB)

    all_results = {}

    for label, ckpt_path, cond_lmdb, eval_beat_windows in [
        ("GT beats model",                    GT_CKPT,  GT_LMDB,  gt_beat_windows),
        ("Extracted beats model (vs GT)",     EXT_CKPT, EXT_LMDB, gt_beat_windows),
        ("Extracted beats model (vs Extracted)", EXT_CKPT, EXT_LMDB, ext_beat_windows),
    ]:
        # Load conditioning windows from cond_lmdb
        env = lmdb.open(cond_lmdb, lock=False, readahead=False)
        windows = []
        with env.begin() as txn:
            for key in txn.cursor().iternext(values=False):
                uid = key.decode("ascii")
                ae  = AudioExample(txn.get(key))
                beat_full = ae.as_dict()["beat"]["data"].astype(np.float32)[0]
                n = beat_full.shape[0]
                for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                    if (uid, start) in eval_beat_windows:
                        windows.append({"uid": uid, "start": start,
                                        "beat": beat_full[start:start+SEQ_LEN]})
        env.close()
        print(f"\nLoaded {label}: {len(windows)} windows")

        model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

        records = run_eval(model, windows, eval_beat_windows, qt, laya_map, device, label)
        all_results[label] = records

    # Save
    def clean(obj):
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.integer): return int(obj)
        return obj

    out = {}
    for label, records in all_results.items():
        out[label] = {}
        for cond in ["gt", "shuf", "zero"]:
            out[label][cond] = [{"cc": r["cc"], "lag": r["lag"], "laya": r["laya"],
                                  "n_onsets": r["n_onsets"], "n_beats": r["n_beats"]}
                                 for r in records[cond]]
    with open("phase_eval_results.json", "w") as f:
        json.dump(clean(out), f, indent=2)
    print(f"\nSaved phase_eval_results.json")


if __name__ == "__main__":
    main()
