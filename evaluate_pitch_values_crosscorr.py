"""
Cross-correlation between raw pitch contour values and beat signal.

Unlike evaluate_contour_crosscorr.py which uses pitch velocity (|Δcents|),
this script correlates the actual pitch values (cents, mean-subtracted)
with the beat pulse. Tests whether absolute pitch level tracks beat positions.

Outputs:
  - pitch_values_crosscorr_results.json
  - pitch_values_crosscorr_plot.png
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import joblib

warnings.filterwarnings("ignore")

FPS     = 100
SEQ_LEN = 1200

BEATS_DIR = "/home/vm2426/HMR_processed/beats"
PITCH_DIR = "/home/vm2426/HMR_processed/pitch"
PRIME_DIR = "outputs_prime"
QT_PATH   = ("/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi"
             "/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95"
             "/diffusion_pitch_model-qt.joblib")

qt = joblib.load(QT_PATH)

TOKEN_TO_CENTS_SCALE  = 10.0
TOKEN_TO_CENTS_OFFSET = -4915.0
SILENCE_TOKEN = 200

VAL_UIDS = [
    "20012", "20018", "20020", "20027", "20037", "20042",
    "21007", "21013", "21042", "22006", "23008",
    "20045", "21040", "21055", "22017", "22019", "23020",
]


def load_beat(uid):
    path = os.path.join(BEATS_DIR, f"{uid}_beat.npy")
    if not os.path.exists(path):
        return None, None
    b = np.load(path)
    return b[0], b[1]


def load_gt_pitch(uid):
    f0_path   = os.path.join(PITCH_DIR, f"{uid}_f0.npy")
    conf_path = os.path.join(PITCH_DIR, f"{uid}_confidence.npy")
    if not os.path.exists(f0_path):
        return None, None
    f0   = np.load(f0_path).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32) if os.path.exists(conf_path) else np.ones_like(f0)
    return f0, conf


def hz_to_cents(f0, conf=None, conf_thr=0.5):
    f0 = f0.copy().astype(np.float32)
    silence = f0 < 50
    if conf is not None:
        silence |= (conf < conf_thr)
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = np.where(~silence, 1200.0 * np.log2(f0 / 440.0), np.nan)
    return cents, silence


def pitch_values_signal(cents, silence):
    """
    Return mean-subtracted pitch values, with silence frames set to 0.
    Unlike velocity, this uses the actual pitch level at each frame.
    """
    sig = cents.copy()
    sig[silence] = np.nan
    # Mean-subtract (voiced frames only) so the signal is centered
    voiced_mean = np.nanmean(sig)
    sig = sig - voiced_mean
    sig = np.nan_to_num(sig, nan=0.0)  # silence → 0 after centering
    return sig.astype(np.float32)


def tokens_to_cents(qt_normalized):
    tokens = qt.inverse_transform(
        qt_normalized.reshape(-1, 1)).flatten().astype(np.float32)
    silence = tokens < SILENCE_TOKEN
    cents = np.where(~silence, tokens * TOKEN_TO_CENTS_SCALE + TOKEN_TO_CENTS_OFFSET, np.nan)
    return cents, silence


def get_beat_period(beat_arr):
    peaks, _ = find_peaks(beat_arr, height=0.5, distance=3)
    if len(peaks) < 2:
        return None
    return float(np.median(np.diff(peaks)))


def normalized_crosscorr(sig_a, sig_b, max_lag):
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    corr_full = np.correlate(a, b, mode='full') / norm
    center = len(corr_full) // 2
    corr = corr_full[center - max_lag: center + max_lag + 1]
    lags  = np.arange(-max_lag, max_lag + 1)
    return corr, lags


# ── 1. GT data ────────────────────────────────────────────────────────────────

print("=" * 70)
print("PART 1: GT pitch VALUES vs GT beats (real data)")
print("=" * 70)

gt_corr_curves   = []
shuf_corr_curves = []
beat_periods_all = []
gt_data = {}

for uid in VAL_UIDS:
    f0, conf = load_gt_pitch(uid)
    beat_arr, _ = load_beat(uid)
    if f0 is None or beat_arr is None:
        continue
    T = min(len(f0), len(beat_arr))
    cents, sil = hz_to_cents(f0[:T], conf[:T])
    sig  = pitch_values_signal(cents, sil)
    beat = beat_arr[:T].astype(np.float32)
    if beat.sum() == 0:
        continue
    bp = get_beat_period(beat)
    if bp is None or bp < 3:
        continue
    gt_data[uid] = (sig, beat, bp)

print(f"Loaded {len(gt_data)} recordings.\n")

MAX_LAG  = 200
all_uids = list(gt_data.keys())

for uid in all_uids:
    sig, beat, bp = gt_data[uid]
    T = min(len(sig), len(beat))
    n_windows = T // SEQ_LEN

    for w in range(n_windows):
        s     = w * SEQ_LEN
        s_win = sig[s:s+SEQ_LEN]
        b_win = beat[s:s+SEQ_LEN]

        bp_w = get_beat_period(b_win)
        if bp_w is None or bp_w < 3:
            continue
        beat_periods_all.append(bp_w)

        corr, lags = normalized_crosscorr(s_win, b_win, MAX_LAG)
        gt_corr_curves.append(corr)

        other_uid = np.random.choice([u for u in all_uids if u != uid])
        _, b_other, _ = gt_data[other_uid]
        T2 = min(len(b_other), s + SEQ_LEN) - s
        b_shuf = b_other[s:s+SEQ_LEN] if T2 >= SEQ_LEN else np.tile(b_other, SEQ_LEN // len(b_other) + 1)[:SEQ_LEN]
        corr_s, _ = normalized_crosscorr(s_win, b_shuf, MAX_LAG)
        shuf_corr_curves.append(corr_s)

mean_beat_period = np.mean(beat_periods_all) if beat_periods_all else 30.0
gt_mean   = np.mean(gt_corr_curves,   axis=0)
shuf_mean = np.mean(shuf_corr_curves, axis=0)
diff_mean = gt_mean - shuf_mean

half_bp = int(mean_beat_period / 2)
center  = MAX_LAG
peak_gt   = gt_mean[center - half_bp: center + half_bp + 1].max()
peak_shuf = shuf_mean[center - half_bp: center + half_bp + 1].max()

print(f"Windows: {len(gt_corr_curves)}")
print(f"Mean beat period: {mean_beat_period:.1f} frames ({mean_beat_period/FPS*1000:.0f} ms)")
print(f"Peak CC (GT):   {peak_gt:.4f}")
print(f"Peak CC (shuf): {peak_shuf:.4f}")
print(f"GT - shuf gap:  {peak_gt - peak_shuf:.4f}")


# ── 2. Generated outputs ──────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("PART 2: Generated pitch VALUES vs GT beats (outputs_prime/)")
print("=" * 70)

PITCH_FILES = {
    "noprime_gt":        "pitch_noprime_gt.npy",
    "prime400_gt":       "pitch_prime400_gt.npy",
    "prime400_shuf":     "pitch_prime400_shuf.npy",
    "beatprime400_gt":   "pitch_beatprime400_gt.npy",
    "beatprime400_shuf": "pitch_beatprime400_shuf.npy",
}

gen_curves = {k: [] for k in PITCH_FILES}

folders = sorted([d for d in os.listdir(PRIME_DIR) if os.path.isdir(os.path.join(PRIME_DIR, d))])

for folder in folders:
    uid = folder.split("_")[0]
    beat_arr, _ = load_beat(uid)
    if beat_arr is None:
        continue

    parts = folder.split("_")
    try:
        start = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        start = 0

    b_win = beat_arr[start: start + SEQ_LEN]
    if len(b_win) < SEQ_LEN:
        b_win = np.pad(b_win, (0, SEQ_LEN - len(b_win)))

    bp = get_beat_period(b_win)
    if bp is None or bp < 3:
        continue

    for key, fname in PITCH_FILES.items():
        fpath = os.path.join(PRIME_DIR, folder, fname)
        if not os.path.exists(fpath):
            continue

        p = np.load(fpath).astype(np.float32)
        if p.ndim > 1:
            p = p.flatten()
        p = p[:SEQ_LEN]
        if len(p) < SEQ_LEN:
            p = np.pad(p, (0, SEQ_LEN - len(p)))

        cents_gen, sil_gen = tokens_to_cents(p)
        sig_gen = pitch_values_signal(cents_gen, sil_gen)

        corr, _ = normalized_crosscorr(sig_gen, b_win, MAX_LAG)
        gen_curves[key].append(corr)

print(f"Folders processed: {len(folders)}")
for key, curves in gen_curves.items():
    if curves:
        mean_curve = np.mean(curves, axis=0)
        peak_val = mean_curve[center - half_bp: center + half_bp + 1].max()
        print(f"  {key:<22}: {len(curves)} windows, peak CC={peak_val:.4f}")


# ── 3. Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lags_ms = lags / FPS * 1000

ax = axes[0]
ax.plot(lags_ms, gt_mean,   label="GT beat",      color="steelblue", lw=2)
ax.plot(lags_ms, shuf_mean, label="Shuffled beat", color="orange",    lw=2, ls="--")
ax.plot(lags_ms, diff_mean, label="GT − shuffled", color="green",     lw=1.5, ls=":")
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
for k in range(-3, 4):
    ax.axvline(k * mean_beat_period / FPS * 1000, color="red", lw=0.6, alpha=0.3, ls=":")
ax.set_title("GT vocal pitch VALUES × GT beats\n(real data)")
ax.set_xlabel("Lag (ms)")
ax.set_ylabel("Normalized cross-correlation")
ax.legend(fontsize=9)
ax.set_xlim(-MAX_LAG / FPS * 1000, MAX_LAG / FPS * 1000)

ax = axes[1]
colors = {"noprime_gt": "steelblue", "prime400_gt": "seagreen",
          "prime400_shuf": "salmon", "beatprime400_gt": "purple",
          "beatprime400_shuf": "orchid"}
for key, curves in gen_curves.items():
    if not curves:
        continue
    mean_curve = np.mean(curves, axis=0)
    ax.plot(lags_ms, mean_curve, label=key, color=colors.get(key, "gray"), lw=1.8)
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
for k in range(-3, 4):
    ax.axvline(k * mean_beat_period / FPS * 1000, color="red", lw=0.6, alpha=0.3, ls=":")
ax.set_title("Generated pitch VALUES × GT beats\n(outputs_prime/)")
ax.set_xlabel("Lag (ms)")
ax.legend(fontsize=8)
ax.set_xlim(-MAX_LAG / FPS * 1000, MAX_LAG / FPS * 1000)

plt.suptitle("Pitch contour VALUES cross-correlation with beat signal", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("pitch_values_crosscorr_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved pitch_values_crosscorr_plot.png")

results = {
    "mean_beat_period_frames": float(mean_beat_period),
    "n_gt_windows":            len(gt_corr_curves),
    "lags_ms":                 lags_ms.tolist(),
    "gt_data": {
        "peak_gt":   float(peak_gt),
        "peak_shuf": float(peak_shuf),
        "gap":       float(peak_gt - peak_shuf),
        "gt_beat_mean_corr":   gt_mean.tolist(),
        "shuf_beat_mean_corr": shuf_mean.tolist(),
    },
    "generated": {
        key: {
            "n_windows": len(curves),
            "mean_corr": np.mean(curves, axis=0).tolist() if curves else [],
            "peak_cc":   float(np.mean(curves, axis=0)[center - half_bp: center + half_bp + 1].max()) if curves else 0.0,
        }
        for key, curves in gen_curves.items()
    },
}

with open("pitch_values_crosscorr_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved pitch_values_crosscorr_results.json")
print("\nDone.")
