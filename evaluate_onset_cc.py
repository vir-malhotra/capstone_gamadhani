"""
Evaluate different onset representations for cross-correlation with beats.

Compares:
  (A) Continuous |Δcents| — current approach
  (B) Binary threshold on |Δcents| — various thresholds
  (C) Stable-note onsets — ISMIR 2021 paper §3 approach:
        segments > 250ms within ±35 cents of a svara location

Runs on:
  - GT real vocal pitch (ground truth) → gives upper bound
  - Generated pitch from outputs_prime/ (noprime_gt, prime400_gt, prime400_shuf)
"""

import os, sys, json, warnings
import numpy as np
from scipy.signal import find_peaks
import joblib

warnings.filterwarnings("ignore")

FPS     = 100
SEQ_LEN = 1200
SILENCE_TOKEN = 200

BEATS_DIR = "/home/vm2426/HMR_processed/beats"
PITCH_DIR = "/home/vm2426/HMR_processed/pitch"
PRIME_DIR = "outputs_prime"
QT_PATH   = ("/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi"
             "/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95"
             "/diffusion_pitch_model-qt.joblib")

qt = joblib.load(QT_PATH)

TOKEN_TO_CENTS_SCALE  = 10.0
TOKEN_TO_CENTS_OFFSET = -4915.0

VAL_UIDS = [
    "20012", "20018", "20020", "20027", "20037", "20042",
    "21007", "21013", "21042", "22006", "23008",
    "20045", "21040", "21055", "22017", "22019", "23020",
]


# ── Pitch loading ─────────────────────────────────────────────────────────────

def load_beat(uid):
    path = os.path.join(BEATS_DIR, f"{uid}_beat.npy")
    if not os.path.exists(path): return None, None
    b = np.load(path)
    return b[0], b[1]


def load_gt_pitch_cents(uid, conf_thr=0.5):
    """Load CREPE f0 → cents, masking silence/low-confidence frames."""
    f0_path   = os.path.join(PITCH_DIR, f"{uid}_f0.npy")
    conf_path = os.path.join(PITCH_DIR, f"{uid}_confidence.npy")
    if not os.path.exists(f0_path): return None, None
    f0   = np.load(f0_path).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32) if os.path.exists(conf_path) else np.ones_like(f0)
    silence = (f0 < 50) | (conf < conf_thr)
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = np.where(~silence, 1200.0 * np.log2(f0 / 440.0), 0.0)
    return cents, silence


def generated_to_cents(qt_arr):
    """QT-normalised generated output → cents + silence mask."""
    tokens = qt.inverse_transform(qt_arr.reshape(-1, 1)).flatten().astype(np.float32)
    silence = tokens < SILENCE_TOKEN
    cents   = np.where(~silence, tokens * TOKEN_TO_CENTS_SCALE + TOKEN_TO_CENTS_OFFSET, 0.0)
    return cents, silence


# ── Onset representations ─────────────────────────────────────────────────────

def onset_continuous_velocity(cents, silence):
    """(A) |Δcents|, zeroed at silence. Current approach."""
    vel = np.abs(np.diff(cents, prepend=cents[0]))
    vel[silence] = 0.0
    return vel.astype(np.float32)


def onset_binary(cents, silence, threshold=50.0):
    """(B) Binary: 1 where |Δcents| >= threshold, else 0."""
    vel = np.abs(np.diff(cents, prepend=cents[0]))
    vel[silence] = 0.0
    return (vel >= threshold).astype(np.float32)


def onset_stable_notes(cents, silence, min_dur_frames=25, cent_tol=35.0, gap_tol_frames=10):
    """
    (C) Stable-note segmentation (ISMIR 2021 §3):
      1. Build long-term pitch histogram (25-cent bins) from voiced frames.
      2. Find svara locations = prominent histogram peaks.
      3. Label 'stable' frames: voiced + within ±cent_tol of nearest svara.
      4. Fill gaps ≤ gap_tol_frames within the same svara.
      5. Keep stable segments ≥ min_dur_frames (250ms at 100Hz).
      6. Onset = 1 at the first frame of each stable segment.

    Returns binary onset signal (1 at note starts, 0 elsewhere).
    """
    voiced = ~silence
    voiced_cents = cents[voiced]

    if len(voiced_cents) < 50:
        return np.zeros(len(cents), dtype=np.float32)

    # 1. Build pitch histogram (25-cent bins, roughly 1 octave = 48 bins)
    # Fold to octave (0-1200 cents) first for robustness
    folded = voiced_cents % 1200.0
    hist, bin_edges = np.histogram(folded, bins=48, range=(0, 1200))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0  # cents within octave

    # 2. Find svara = peaks in histogram with prominence
    peaks, props = find_peaks(hist, height=hist.max() * 0.05, distance=2)
    if len(peaks) == 0:
        return np.zeros(len(cents), dtype=np.float32)
    svara_cents_folded = bin_centers[peaks]  # cents within octave

    # 3. For each voiced frame, find nearest svara (in octave-folded space)
    folded_all = cents % 1200.0
    stable = np.zeros(len(cents), dtype=bool)
    for sc in svara_cents_folded:
        dist = np.abs(folded_all - sc)
        # wrap-around distance
        dist = np.minimum(dist, 1200.0 - dist)
        stable |= (voiced & (dist <= cent_tol))

    # 4. Fill short gaps within stable regions (≤ gap_tol_frames)
    filled = stable.copy()
    in_gap = False
    gap_start = 0
    for i in range(len(filled)):
        if stable[i]:
            if in_gap:
                gap_len = i - gap_start
                if gap_len <= gap_tol_frames:
                    filled[gap_start:i] = True
            in_gap = False
        else:
            if not in_gap:
                gap_start = i
                in_gap = True

    # 5. Find stable segments ≥ min_dur_frames and mark onsets
    onsets = np.zeros(len(cents), dtype=np.float32)
    in_seg = False
    seg_start = 0
    for i in range(len(filled) + 1):
        cur = filled[i] if i < len(filled) else False
        if cur and not in_seg:
            in_seg = True
            seg_start = i
        elif not cur and in_seg:
            seg_len = i - seg_start
            if seg_len >= min_dur_frames:
                onsets[seg_start] = 1.0
            in_seg = False

    return onsets


# ── Cross-correlation ─────────────────────────────────────────────────────────

def normalized_crosscorr(sig_a, sig_b, max_lag=200):
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    corr_full = np.correlate(a, b, mode='full') / norm
    center = len(corr_full) // 2
    return corr_full[center - max_lag: center + max_lag + 1]


def get_beat_period(beat_arr):
    peaks, _ = find_peaks(beat_arr, height=0.5, distance=3)
    if len(peaks) < 2: return None
    return float(np.median(np.diff(peaks)))


def peak_cc_in_window(corr, beat_period, max_lag=200):
    """Return max CC within ±half-beat-period around lag=0."""
    half_bp = int(beat_period / 2)
    center  = max_lag
    return corr[center - half_bp: center + half_bp + 1].max()


# ── Evaluation loop ───────────────────────────────────────────────────────────

MAX_LAG = 200
BINARY_THRESHOLDS = [25, 50, 100, 150, 200]  # cents

def evaluate_representations(cents_list, beat_list, label="", shuf_beat_list=None):
    """
    Given parallel lists of (cents, silence) and beat_arr,
    compute peak CC for each onset representation, averaged over windows.
    If shuf_beat_list is given, also prints GT−shuf gap.
    """
    results = {
        "continuous": [],
        **{f"binary_{t}c": [] for t in BINARY_THRESHOLDS},
        "stable_notes": [],
    }
    results_shuf = {k: [] for k in results} if shuf_beat_list is not None else None
    beat_periods = []

    for idx, ((cents, silence), beat_arr) in enumerate(zip(cents_list, beat_list)):
        bp = get_beat_period(beat_arr)
        if bp is None or bp < 3: continue
        beat_periods.append(bp)

        vel_cont  = onset_continuous_velocity(cents, silence)
        sn_onsets = onset_stable_notes(cents, silence)

        corr_cont = normalized_crosscorr(vel_cont, beat_arr, MAX_LAG)
        corr_sn   = normalized_crosscorr(sn_onsets, beat_arr, MAX_LAG)

        results["continuous"].append(peak_cc_in_window(corr_cont, bp, MAX_LAG))
        results["stable_notes"].append(peak_cc_in_window(corr_sn, bp, MAX_LAG))

        for thr in BINARY_THRESHOLDS:
            bin_sig = onset_binary(cents, silence, threshold=thr)
            corr_b  = normalized_crosscorr(bin_sig, beat_arr, MAX_LAG)
            results[f"binary_{thr}c"].append(peak_cc_in_window(corr_b, bp, MAX_LAG))

        if shuf_beat_list is not None:
            beat_shuf = shuf_beat_list[idx]
            corr_cont_s = normalized_crosscorr(vel_cont, beat_shuf, MAX_LAG)
            corr_sn_s   = normalized_crosscorr(sn_onsets, beat_shuf, MAX_LAG)
            results_shuf["continuous"].append(peak_cc_in_window(corr_cont_s, bp, MAX_LAG))
            results_shuf["stable_notes"].append(peak_cc_in_window(corr_sn_s, bp, MAX_LAG))
            for thr in BINARY_THRESHOLDS:
                bin_sig = onset_binary(cents, silence, threshold=thr)
                corr_b  = normalized_crosscorr(bin_sig, beat_shuf, MAX_LAG)
                results_shuf[f"binary_{thr}c"].append(peak_cc_in_window(corr_b, bp, MAX_LAG))

    mean_bp = np.mean(beat_periods) if beat_periods else 0
    n = len(results['continuous'])
    print(f"\n{label} ({n} windows, mean beat period={mean_bp:.0f}fr / {mean_bp/FPS*1000:.0f}ms)")
    if results_shuf is not None:
        print(f"  {'Method':<20} {'GT CC':>8} {'Shuf CC':>8} {'Gap':>8}")
        print(f"  {'-'*46}")
        for name in results:
            gt_v   = np.mean(results[name])      if results[name]      else 0
            shuf_v = np.mean(results_shuf[name]) if results_shuf[name] else 0
            print(f"  {name:<20} {gt_v:>8.4f} {shuf_v:>8.4f} {gt_v-shuf_v:>+8.4f}")
    else:
        print(f"  {'Method':<20} {'Peak CC':>8}")
        print(f"  {'-'*30}")
        for name, vals in results.items():
            if vals:
                print(f"  {name:<20} {np.mean(vals):>8.4f}")
    return results


# ── 1. GT real vocal ──────────────────────────────────────────────────────────
print("=" * 60)
print("1. GT REAL VOCAL pitch → GT beats")
print("=" * 60)

gt_pairs = []
gt_beats = []
for uid in VAL_UIDS:
    cents, sil = load_gt_pitch_cents(uid)
    beat_arr, _ = load_beat(uid)
    if cents is None or beat_arr is None: continue
    T = min(len(cents), len(beat_arr))
    for w in range(T // SEQ_LEN):
        s = w * SEQ_LEN
        gt_pairs.append((cents[s:s+SEQ_LEN], sil[s:s+SEQ_LEN]))
        gt_beats.append(beat_arr[s:s+SEQ_LEN])

gt_results = evaluate_representations(gt_pairs, gt_beats, "GT real vocal")


# ── 2. Generated pitch (outputs_prime/) ──────────────────────────────────────
print("\n" + "=" * 60)
print("2. GENERATED pitch (outputs_prime/) → GT beats")
print("=" * 60)

PITCH_FILES = {
    "noprime_gt":    ("pitch_noprime_gt.npy",   0),
    "prime400_gt":   ("pitch_prime400_gt.npy",  400),
    "prime400_shuf": ("pitch_prime400_shuf.npy",400),
}

for key, (fname, prime_len) in PITCH_FILES.items():
    gen_pairs = []
    gen_beats = []

    folders = sorted([
        d for d in os.listdir(PRIME_DIR)
        if os.path.isdir(os.path.join(PRIME_DIR, d))
    ])
    for folder in folders:
        uid = folder.split("_")[0]
        beat_arr, _ = load_beat(uid)
        if beat_arr is None: continue

        fpath = os.path.join(PRIME_DIR, folder, fname)
        if not os.path.exists(fpath): continue

        p = np.load(fpath).astype(np.float32).flatten()[:SEQ_LEN]
        if len(p) < SEQ_LEN: p = np.pad(p, (0, SEQ_LEN - len(p)))

        cents, sil = generated_to_cents(p)
        parts = folder.split("_")
        try: start = int(parts[1]) if len(parts) > 1 else 0
        except ValueError: start = 0

        b_win = beat_arr[start: start + SEQ_LEN]
        if len(b_win) < SEQ_LEN: b_win = np.pad(b_win, (0, SEQ_LEN - len(b_win)))

        # Exclude prime frames from evaluation
        gen_pairs.append((cents[prime_len:], sil[prime_len:]))
        gen_beats.append(b_win[prime_len:])

    # Build shuffled beats (different recording's beat, same offset)
    n = len(gen_beats)
    shuf_beats = [gen_beats[(i + n // 2) % n] for i in range(n)]

    evaluate_representations(gen_pairs, gen_beats, key, shuf_beat_list=shuf_beats)


print("\nDone.")
