"""
Proper beat adherence evaluation following Sketch2Sound / Music ControlNet approach.

For each generated audio file (gt, shuf, zero conditions):
  1. Run madmom beat tracker on the generated WAV
  2. Compare detected beat times to the input GT beat times
  3. Compute F1 with ±70ms tolerance (standard mir_eval beat metric)

Uses the 22 audio files already in outputs/ (11 val recordings × 2 windows × 3 conditions).
"""

import os, sys, json
import numpy as np
import mir_eval

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.src.hmr_dataset import _extract_events

OUT_DIR     = "outputs"
SAMPLE_RATE = 16000
TOLERANCE   = 0.07   # ±70ms — standard mir_eval beat tracking tolerance


# ── Beat tracker ──────────────────────────────────────────────────────────────

def track_beats(wav_path: str) -> np.ndarray:
    """Run madmom DBNBeatTrackingProcessor on a WAV file. Returns beat times in seconds."""
    import madmom
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act  = madmom.features.beats.RNNBeatProcessor()(wav_path)
    beats = proc(act)
    return np.array(beats, dtype=float)


# ── GT beat times from saved .npy ─────────────────────────────────────────────

def get_gt_beat_times(beat_npy_path: str) -> np.ndarray:
    """Extract GT beat event times (seconds) from the saved beat.npy signal."""
    beat = np.load(beat_npy_path)
    events = _extract_events(beat)  # frame indices at 100Hz
    return events / 100.0           # → seconds


# ── mir_eval beat F1 ─────────────────────────────────────────────────────────

def beat_f1(ref_times: np.ndarray, est_times: np.ndarray, tol: float = TOLERANCE):
    """Standard mir_eval beat F1 with ±tol tolerance window."""
    if len(ref_times) == 0 or len(est_times) == 0:
        return dict(f1=0., precision=0., recall=0., n_ref=len(ref_times), n_est=len(est_times))
    # Use mir_eval matching to get TP count
    ref_times = mir_eval.util.adjust_events(ref_times)[0]
    est_times = mir_eval.util.adjust_events(est_times)[0]
    matching  = mir_eval.util.match_events(ref_times, est_times, tol)
    tp        = len(matching)
    precision = tp / len(est_times) if len(est_times) > 0 else 0.
    recall    = tp / len(ref_times) if len(ref_times) > 0 else 0.
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.
    return dict(f1=f1, precision=precision, recall=recall,
                n_ref=len(ref_times), n_est=len(est_times))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    folders = sorted(os.listdir(OUT_DIR))
    results = {cond: [] for cond in ["gt", "shuf", "zero"]}
    rows    = []

    print(f"{'Folder':<18}  #GT  {'GT_F1':>6} {'SH_F1':>6} {'ZR_F1':>6}  {'GT_est':>6} {'ZR_est':>6}")
    print("-" * 70)

    for folder in folders:
        d = os.path.join(OUT_DIR, folder)
        beat_path = os.path.join(d, "beat.npy")
        if not os.path.exists(beat_path):
            continue

        gt_times = get_gt_beat_times(beat_path)
        row = {"folder": folder, "n_gt": len(gt_times)}

        for cond in ["gt", "shuf", "zero"]:
            wav_path = os.path.join(d, f"audio_{cond}.wav")
            if not os.path.exists(wav_path):
                row[cond] = None
                continue
            est_times = track_beats(wav_path)
            scores    = beat_f1(gt_times, est_times)
            row[cond] = scores
            results[cond].append(scores)

        rows.append(row)
        gt_f1  = row["gt"]["f1"]   if row["gt"]   else 0.
        sh_f1  = row["shuf"]["f1"] if row["shuf"] else 0.
        zr_f1  = row["zero"]["f1"] if row["zero"] else 0.
        gt_est = row["gt"]["n_est"]   if row["gt"]   else 0
        zr_est = row["zero"]["n_est"] if row["zero"] else 0
        print(f"{folder:<18}  {len(gt_times):3d}  {gt_f1:>6.3f} {sh_f1:>6.3f} {zr_f1:>6.3f}  {gt_est:>6} {zr_est:>6}")

    print("=" * 70)
    for cond, label in [("gt", "GT conditioned"), ("shuf", "Shuffled beat"), ("zero", "Unconditioned")]:
        if not results[cond]: continue
        f1s  = [r["f1"]       for r in results[cond]]
        prec = [r["precision"] for r in results[cond]]
        rec  = [r["recall"]    for r in results[cond]]
        nest = [r["n_est"]     for r in results[cond]]
        print(f"{label:<20}  F1={np.mean(f1s):.3f}  Prec={np.mean(prec):.3f}  Rec={np.mean(rec):.3f}  Est_beats={np.mean(nest):.1f}")

    with open("beat_tracker_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print("\nSaved beat_tracker_results.json")


if __name__ == "__main__":
    main()
