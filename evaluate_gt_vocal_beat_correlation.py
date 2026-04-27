"""
GT vocal onset vs GT beat onset correlation.

Tests the hypothesis that vocal phrase onsets are correlated with beat positions
using the actual ground-truth data — no model generation needed.

Three conditions per window:
  1. GT beats      — F1 between vocal onsets and the actual annotated beat frames
  2. Shuffled beats — F1 between vocal onsets and a different recording's beats
  3. Random beats   — F1 between vocal onsets and uniform-random beat positions

If the hypothesis holds: GT >> shuffled ≈ random.

Also reports:
  - Breakdown by taal (Teentaal / Ektaal / Jhaptaal / Rupak)
  - Breakdown by laya (Vilambit / Madhya / Drut)
  - Onset stats (avg onsets per window, avg beats per window)
"""

import os, sys, json, random
import numpy as np
import lmdb
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.src.protobuf.data_example import AudioExample
from hmr_dataset import _normalize_pitch

import joblib
import pandas as pd

HMR_DIR    = "/home/vm2426/HMR dataset"
PROC_DIR   = "/home/vm2426/HMR_processed"
LMDB_TRAIN = os.path.join(PROC_DIR, "lmdb", "train")
LMDB_VAL   = os.path.join(PROC_DIR, "lmdb", "val")
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CSV_PATH   = "/home/vm2426/HMDf.xlsx - HMDf.csv"

SEQ_LEN      = 1200   # 12s at 100Hz
ONSET_WINDOW = 5      # ±5 frames = ±50ms
ONSET_JUMP_THR = 10   # token-space jump ≥10 counts as onset


# ── Onset detection (same as evaluate_beat_alignment.py) ─────────────────────

def detect_onsets(norm_pitch: np.ndarray, qt) -> np.ndarray:
    """Detect note onsets from QT-normalised pitch tokens."""
    tokens = qt.inverse_transform(norm_pitch.reshape(-1, 1)).flatten()
    silence = tokens < 200
    sil_to_voiced = np.where(silence[:-1] & ~silence[1:])[0] + 1
    jumps = np.abs(np.diff(tokens))
    big_jump = (jumps >= ONSET_JUMP_THR) & ~silence[1:]
    jump_onsets = np.where(big_jump)[0] + 1
    return np.unique(np.concatenate([sil_to_voiced, jump_onsets]))


# ── Beat alignment F1 ─────────────────────────────────────────────────────────

def beat_f1(onsets: np.ndarray, beat_frames: np.ndarray,
            window: int = ONSET_WINDOW) -> dict:
    if len(onsets) == 0 or len(beat_frames) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "n_onsets": len(onsets), "n_beats": len(beat_frames)}
    tp = 0
    matched_onsets = set()
    for b in beat_frames:
        near = np.where(np.abs(onsets - b) <= window)[0]
        if len(near) > 0:
            tp += 1
            matched_onsets.add(near[0])
    fp = len(onsets) - len(matched_onsets)
    fn = len(beat_frames) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec,
            "n_onsets": len(onsets), "n_beats": len(beat_frames)}


# ── Load all windows from both splits ─────────────────────────────────────────

def load_all_windows(qt):
    windows = []
    for split_path in [LMDB_TRAIN, LMDB_VAL]:
        env = lmdb.open(split_path, lock=False, readahead=False)
        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            for key in keys:
                ae = AudioExample(txn.get(key))
                d  = ae.as_dict()

                f0   = d["pitch"]["data"].astype(np.float32)
                beat = d["beat"]["data"].astype(np.float32)   # [3, N]

                uid = key.decode("ascii")
                n   = min(f0.shape[0], beat.shape[1])
                f0, beat = f0[:n], beat[:, :n]

                # Take non-overlapping 12s windows
                for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                    f0_win   = f0[start : start + SEQ_LEN]
                    beat_win = beat[0, start : start + SEQ_LEN]  # pulse channel

                    norm = _normalize_pitch(f0_win, qt)
                    onsets = detect_onsets(norm, qt)
                    # Find beat peaks: first frame of each contiguous run where beat > 0.5
                    # (the pulse is smoothed to a 5-frame box, so collapse runs to single positions)
                    above = (beat_win > 0.5).astype(int)
                    rises = np.where(np.diff(np.concatenate([[0], above])) == 1)[0]
                    beat_frames = rises

                    windows.append({
                        "uid":         uid,
                        "start":       start,
                        "onsets":      onsets,
                        "beat_frames": beat_frames,
                        "n_onsets":    len(onsets),
                        "n_beats":     len(beat_frames),
                    })
        env.close()
    return windows


# ── Metadata lookup ───────────────────────────────────────────────────────────

def load_metadata():
    df = pd.read_csv(CSV_PATH)
    df["UID"] = df["UID"].astype(str).str.strip()
    meta = {}
    for _, row in df.iterrows():
        uid = row["UID"]
        meta[uid] = {
            "taal":  str(row.get("Taal", "")).strip().lower(),
            "laya":  str(row.get("Corrected Lay Label", "")).strip(),
            "name":  str(row.get("Name", uid)).strip(),
        }
    return meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    qt = joblib.load(QT_PATH)
    print("Loading all windows from LMDB...", flush=True)
    windows = load_all_windows(qt)
    print(f"  {len(windows)} windows from {len(set(w['uid'] for w in windows))} recordings")

    meta = load_metadata()

    # Collect all beat_frames arrays for shuffling
    all_beat_frames = [w["beat_frames"] for w in windows]

    results = defaultdict(list)   # uid → list of result dicts

    print("\nComputing correlations...", flush=True)
    for i, w in enumerate(windows):
        onsets = w["onsets"]
        bf_gt  = w["beat_frames"]

        # GT beats
        r_gt = beat_f1(onsets, bf_gt)

        # Shuffled beats — pick a window from a different recording
        j = i
        while windows[j]["uid"] == w["uid"]:
            j = random.randint(0, len(windows) - 1)
        r_shuf = beat_f1(onsets, windows[j]["beat_frames"])

        # Random beats — same count as GT, uniformly random positions
        if len(bf_gt) > 0:
            rng_frames = np.sort(np.random.choice(SEQ_LEN, size=len(bf_gt), replace=False))
        else:
            rng_frames = np.array([], dtype=int)
        r_rand = beat_f1(onsets, rng_frames)

        results[w["uid"]].append({
            "start":  w["start"],
            "gt":     r_gt,
            "shuf":   r_shuf,
            "rand":   r_rand,
            "n_onsets": len(onsets),
            "n_beats":  len(bf_gt),
        })

    # ── Aggregate ──────────────────────────────────────────────────────────────
    all_gt, all_shuf, all_rand = [], [], []
    for uid_rows in results.values():
        for r in uid_rows:
            all_gt.append(r["gt"]["f1"])
            all_shuf.append(r["shuf"]["f1"])
            all_rand.append(r["rand"]["f1"])

    print(f"\n{'='*60}")
    print("GT VOCAL ONSET vs BEAT CORRELATION")
    print(f"{'='*60}")
    print(f"  Windows: {len(all_gt)}  |  Recordings: {len(results)}")
    print(f"\n{'Condition':<25} {'Mean F1':>8} {'Median':>8} {'Std':>6}")
    print(f"{'-'*50}")
    for label, vals in [("GT beats", all_gt), ("Shuffled beats", all_shuf), ("Random beats", all_rand)]:
        print(f"  {label:<23} {np.mean(vals):>8.3f} {np.median(vals):>8.3f} {np.std(vals):>6.3f}")

    # ── Per-taal breakdown ────────────────────────────────────────────────────
    taal_gt = defaultdict(list)
    taal_shuf = defaultdict(list)
    for uid, uid_rows in results.items():
        taal = meta.get(uid, {}).get("taal", "unknown")
        for r in uid_rows:
            taal_gt[taal].append(r["gt"]["f1"])
            taal_shuf[taal].append(r["shuf"]["f1"])

    print(f"\n{'─'*60}")
    print(f"BY TAAL:")
    print(f"{'Taal':<16} {'N':>5} {'GT F1':>8} {'Shuf F1':>8} {'Gap':>7}")
    print(f"{'─'*50}")
    for taal in sorted(taal_gt):
        gt_m   = np.mean(taal_gt[taal])
        shuf_m = np.mean(taal_shuf[taal])
        print(f"  {taal:<14} {len(taal_gt[taal]):>5} {gt_m:>8.3f} {shuf_m:>8.3f} {gt_m-shuf_m:>+7.3f}")

    # ── Per-laya breakdown ────────────────────────────────────────────────────
    laya_gt = defaultdict(list)
    laya_shuf = defaultdict(list)
    for uid, uid_rows in results.items():
        laya = meta.get(uid, {}).get("laya", "unknown")
        for r in uid_rows:
            laya_gt[laya].append(r["gt"]["f1"])
            laya_shuf[laya].append(r["shuf"]["f1"])

    print(f"\nBY LAYA:")
    print(f"{'Laya':<16} {'N':>5} {'GT F1':>8} {'Shuf F1':>8} {'Gap':>7}")
    print(f"{'─'*50}")
    for laya in sorted(laya_gt):
        gt_m   = np.mean(laya_gt[laya])
        shuf_m = np.mean(laya_shuf[laya])
        print(f"  {laya:<14} {len(laya_gt[laya]):>5} {gt_m:>8.3f} {shuf_m:>8.3f} {gt_m-shuf_m:>+7.3f}")

    # ── Onset stats ───────────────────────────────────────────────────────────
    all_n_onsets = [r["n_onsets"] for uid_rows in results.values() for r in uid_rows]
    all_n_beats  = [r["n_beats"]  for uid_rows in results.values() for r in uid_rows]
    print(f"\nONSET STATS:")
    print(f"  Avg onsets per 12s window: {np.mean(all_n_onsets):.1f}  (median {np.median(all_n_onsets):.1f})")
    print(f"  Avg beats  per 12s window: {np.mean(all_n_beats):.1f}  (median {np.median(all_n_beats):.1f})")
    windows_no_onsets = sum(1 for n in all_n_onsets if n == 0)
    windows_no_beats  = sum(1 for n in all_n_beats  if n == 0)
    print(f"  Windows with 0 onsets: {windows_no_onsets}/{len(all_n_onsets)}")
    print(f"  Windows with 0 beats:  {windows_no_beats}/{len(all_n_beats)}")
    print(f"{'='*60}")

    # Save
    out = {uid: rows for uid, rows in results.items()}
    # convert numpy ints to regular ints for JSON
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open("gt_vocal_beat_correlation.json", "w") as f:
        json.dump(to_serializable(out), f, indent=2)
    print("\nSaved gt_vocal_beat_correlation.json")


if __name__ == "__main__":
    main()
