"""
Evaluate beat alignment for prime-conditioned generations.

Compares across conditions:
  - noprime_gt       : GT beat, no prime
  - prime100_gt      : GT beat + 1s prime
  - prime200_gt      : GT beat + 2s prime
  - prime400_gt      : GT beat + 4s prime
  - prime100_shuf    : shuffled beat + 1s prime  (control)
  - prime200_shuf    : shuffled beat + 2s prime
  - prime400_shuf    : shuffled beat + 4s prime

Beat frames come from the GT beat signal stored in the LMDB.
"""

import os, sys, json, glob
import numpy as np
import lmdb
import joblib
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.src.protobuf.data_example import AudioExample
from hmr_dataset import _normalize_pitch

QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
LMDB_VAL   = "/home/vm2426/HMR_processed/lmdb/val"
OUT_DIR    = "outputs_prime"
SEQ_LEN    = 1200
ONSET_WINDOW   = 5
ONSET_JUMP_THR = 10
PRIME_LENS = [100, 200, 400]
PRIME_TYPES = ["pitch", "beat"]   # pitch prime vs beat-encoded prime


def detect_onsets(norm_pitch, qt):
    tokens = qt.inverse_transform(norm_pitch.reshape(-1, 1)).flatten()
    silence = tokens < 200
    sil_to_voiced = np.where(silence[:-1] & ~silence[1:])[0] + 1
    jumps = np.abs(np.diff(tokens))
    big_jump = (jumps >= ONSET_JUMP_THR) & ~silence[1:]
    jump_onsets = np.where(big_jump)[0] + 1
    return np.unique(np.concatenate([sil_to_voiced, jump_onsets]))


def beat_f1(onsets, beat_frames, window=ONSET_WINDOW):
    if len(onsets) == 0 or len(beat_frames) == 0:
        return 0.0
    tp, matched = 0, set()
    for b in beat_frames:
        near = np.where(np.abs(onsets - b) <= window)[0]
        if len(near) > 0:
            tp += 1; matched.add(near[0])
    prec = tp / len(onsets) if len(onsets) > 0 else 0
    rec  = tp / len(beat_frames) if len(beat_frames) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def main():
    qt = joblib.load(QT_PATH)

    # Load GT beat frames for each val window from LMDB
    beat_map = {}   # uid_start -> beat_frames array
    env = lmdb.open(LMDB_VAL, lock=False, readahead=False)
    with env.begin() as txn:
        for key in txn.cursor().iternext(values=False):
            ae  = AudioExample(txn.get(key))
            d   = ae.as_dict()
            uid = key.decode("ascii")
            beat_full = d["beat"]["data"].astype(np.float32)  # [3, N]
            pulse = beat_full[0]  # channel 0
            n = pulse.shape[0]
            for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                b = pulse[start:start + SEQ_LEN]
                rises = np.where(np.diff(np.concatenate([[0], (b > 0.5).astype(int)])) == 1)[0]
                beat_map[f"{uid}_{start}"] = rises
    env.close()

    # Conditions to evaluate
    conditions = (
        ["noprime_gt"] +
        [f"prime{P}_{cond}"     for P in PRIME_LENS for cond in ["gt", "shuf"]] +
        [f"beatprime{P}_{cond}" for P in PRIME_LENS for cond in ["gt", "shuf"]]
    )

    results = {c: [] for c in conditions}

    folders = sorted(glob.glob(f"{OUT_DIR}/*/"))
    print(f"Evaluating {len(folders)} windows...")

    for folder in folders:
        key = os.path.basename(folder.rstrip("/"))
        beat_frames = beat_map.get(key)
        if beat_frames is None or len(beat_frames) == 0:
            continue

        for cond in conditions:
            fpath = os.path.join(folder, f"pitch_{cond}.npy")
            if not os.path.exists(fpath):
                continue
            pitch = np.load(fpath)

            # Determine prime length for this condition (0 for noprime)
            P = 0
            for pl in PRIME_LENS:
                if f"prime{pl}" in cond:
                    P = pl
                    break

            # Evaluate only the continuation (frames P onwards)
            # so the clamped prime region doesn't trivially inflate F1
            pitch_cont = pitch[P:]
            bf_cont    = beat_frames[beat_frames >= P] - P

            onsets = detect_onsets(pitch_cont, qt)
            f1 = beat_f1(onsets, bf_cont)
            results[cond].append(f1)

    # ── Print results ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PRIME GENERATION — BEAT ALIGNMENT F1")
    print(f"{'='*60}")
    print(f"{'Condition':<22} {'N':>5} {'Mean F1':>8} {'Median':>8} {'Std':>6}")
    print(f"{'-'*55}")

    groups = [
        ("── No prime ───────────────", ["noprime_gt"]),
        ("── Pitch prime 1s (100fr) ─", ["prime100_gt",     "prime100_shuf"]),
        ("── Pitch prime 2s (200fr) ─", ["prime200_gt",     "prime200_shuf"]),
        ("── Pitch prime 4s (400fr) ─", ["prime400_gt",     "prime400_shuf"]),
        ("── Beat prime 1s (100fr) ──", ["beatprime100_gt", "beatprime100_shuf"]),
        ("── Beat prime 2s (200fr) ──", ["beatprime200_gt", "beatprime200_shuf"]),
        ("── Beat prime 4s (400fr) ──", ["beatprime400_gt", "beatprime400_shuf"]),
    ]
    for group_label, conds in groups:
        print(f"\n{group_label}")
        for cond in conds:
            v = results[cond]
            if not v: continue
            tag = "[GT beat]" if cond.endswith("_gt") else "[shuffled]"
            print(f"  {cond:<26} {len(v):>5} {np.mean(v):>8.3f} {np.median(v):>8.3f} {np.std(v):>6.3f}  {tag}")

    # Summary table
    print(f"\n{'─'*65}")
    print("GT beat — effect of prime type and length (continuation only):")
    print(f"  {'Condition':<26} {'Mean F1':>8} {'vs baseline':>12} {'GT-shuf gap':>12}")
    base = np.mean(results["noprime_gt"])
    rows = ["noprime_gt"] + [f"{pt}prime{P}_gt" for pt in ["", "beat"] for P in PRIME_LENS]
    for cond in rows:
        v = results[cond]
        if not v: continue
        shuf_key = cond.replace("_gt", "_shuf")
        shuf_v = results.get(shuf_key, [])
        diff = np.mean(v) - base
        gap  = np.mean(v) - np.mean(shuf_v) if shuf_v else float("nan")
        diff_s = f"{diff:+.3f}" if cond != "noprime_gt" else "(baseline)"
        gap_s  = f"{gap:+.3f}"  if shuf_v else "    n/a"
        print(f"  {cond:<26} {np.mean(v):>8.3f} {diff_s:>12} {gap_s:>12}")

    print(f"{'='*60}")

    with open("prime_eval_results.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f, indent=2)
    print("Saved prime_eval_results.json")


if __name__ == "__main__":
    main()
