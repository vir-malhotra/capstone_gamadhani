import os, sys, json, warnings
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
    """Load 3-channel beat signal, return (beat_pulse, sam_pulse) at 100Hz."""
    path = os.path.join(BEATS_DIR, f"{uid}_beat.npy")
    if not os.path.exists(path):
        return None, None
    b = np.load(path)  # (3, T)
    return b[0], b[1]


def load_gt_pitch(uid):
    """Load CREPE pitch (Hz) and confidence at 100Hz."""
    f0_path   = os.path.join(PITCH_DIR, f"{uid}_f0.npy")
    conf_path = os.path.join(PITCH_DIR, f"{uid}_confidence.npy")
    if not os.path.exists(f0_path):
        return None, None
    f0   = np.load(f0_path).astype(np.float32)
    conf = np.load(conf_path).astype(np.float32) if os.path.exists(conf_path) else np.ones_like(f0)
    return f0, conf


def hz_to_cents(f0, conf=None, conf_thr=0.5):
    """Convert raw Hz f0 to cents (440Hz ref), masking silence and low confidence."""
    f0 = f0.copy().astype(np.float32)
    silence = f0 < 50
    if conf is not None:
        silence |= (conf < conf_thr)
    with np.errstate(divide='ignore', invalid='ignore'):
        cents = np.where(~silence, 1200.0 * np.log2(f0 / 440.0), 0.0)
    return cents, silence


def tokens_to_cents(qt_normalized):
    """Convert QT-normalized pitch (generated output) to cents.
    Pipeline: qt.inverse_transform → token → token*10 - 4915 = cents.
    Silence = token < SILENCE_TOKEN → 0.
    """
    tokens = qt.inverse_transform(
        qt_normalized.reshape(-1, 1)).flatten().astype(np.float32)
    silence = tokens < SILENCE_TOKEN
    cents = np.where(~silence, tokens * TOKEN_TO_CENTS_SCALE + TOKEN_TO_CENTS_OFFSET, 0.0)
    return cents, silence


def cents_to_velocity(cents, silence):
    """Velocity = |Δcents|, zeroed at silence frames."""
    vel = np.abs(np.diff(cents, prepend=cents[0]))
    vel[silence] = 0.0
    return vel.astype(np
    .float32)


def beat_pulse(beat_arr, beat_sr=FPS):
    """Return beat_arr as-is (already a pulse signal at 100Hz)."""
    return beat_arr.astype(np.float32)


def get_beat_period(beat_arr):
    """Estimate beat period in frames from beat pulse."""
    peaks, _ = find_peaks(beat_arr, height=0.5, distance=3)
    if len(peaks) < 2:
        return None
    return float(np.median(np.diff(peaks)))


def normalized_crosscorr(sig_a, sig_b, max_lag):
    """
    Normalized cross-correlation between sig_a and sig_b,
    at lags in [-max_lag, +max_lag].
    Returns (corr_values, lags_array).
    """
    a = sig_a - sig_a.mean()
    b = sig_b - sig_b.mean()
    norm = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    corr_full = np.correlate(a, b, mode='full') / norm
    center = len(corr_full) // 2
    corr = corr_full[center - max_lag: center + max_lag + 1]
    lags  = np.arange(-max_lag, max_lag + 1)
    return corr, lags



print("=" * 70)
print("PART 1: GT pitch velocity vs GT beats (real data)")
print("=" * 70)

gt_corr_curves   = []
shuf_corr_curves = []
beat_periods_all = []

gt_uids_loaded = []
gt_data = {}

for uid in VAL_UIDS:
    f0, conf = load_gt_pitch(uid)
    beat_arr, _ = load_beat(uid)
    if f0 is None or beat_arr is None:
        continue
    T = min(len(f0), len(beat_arr))
    cents, sil = hz_to_cents(f0[:T], conf[:T])
    vel  = cents_to_velocity(cents, sil)
    beat = beat_pulse(beat_arr[:T])
    if vel.sum() == 0 or beat.sum() == 0:
        continue
    bp = get_beat_period(beat)
    if bp is None or bp < 3:
        continue
    gt_data[uid] = (vel, beat, bp)
    gt_uids_loaded.append(uid)

print(f"Loaded {len(gt_data)} recordings.\n")

MAX_LAG = 200

all_uids = list(gt_data.keys())
for uid in all_uids:
    vel, beat, bp = gt_data[uid]

    T = min(len(vel), len(beat))
    n_windows = T // SEQ_LEN

    for w in range(n_windows):
        s = w * SEQ_LEN
        v_win = vel[s:s+SEQ_LEN]
        b_win = beat[s:s+SEQ_LEN]

        bp_w = get_beat_period(b_win)
        if bp_w is None or bp_w < 3:
            continue
        beat_periods_all.append(bp_w)

        corr, lags = normalized_crosscorr(v_win, b_win, MAX_LAG)
        gt_corr_curves.append(corr)

        other_uid = np.random.choice([u for u in all_uids if u != uid])
        _, b_other, _ = gt_data[other_uid]
        T2 = min(len(b_other), s + SEQ_LEN) - s
        if T2 < SEQ_LEN:
            b_shuf = np.tile(b_other, SEQ_LEN // len(b_other) + 1)[:SEQ_LEN]
        else:
            b_shuf = b_other[s:s+SEQ_LEN]
        corr_s, _ = normalized_crosscorr(v_win, b_shuf, MAX_LAG)
        shuf_corr_curves.append(corr_s)

mean_beat_period = np.mean(beat_periods_all) if beat_periods_all else 30.0

print(f"Windows: {len(gt_corr_curves)}")
print(f"Mean beat period: {mean_beat_period:.1f} frames ({mean_beat_period/FPS*1000:.0f} ms)")

gt_mean   = np.mean(gt_corr_curves, axis=0)
shuf_mean = np.mean(shuf_corr_curves, axis=0)
diff_mean = gt_mean - shuf_mean

half_bp = int(mean_beat_period / 2)
center = MAX_LAG
peak_gt   = gt_mean[center - half_bp: center + half_bp + 1].max()
peak_shuf = shuf_mean[center - half_bp: center + half_bp + 1].max()
print(f"Peak CC (GT):  {peak_gt:.4f}")
print(f"Peak CC (shuf):{peak_shuf:.4f}")
print(f"GT - shuf gap: {peak_gt - peak_shuf:.4f}")


print("\n" + "=" * 70)
print("PART 2: Generated pitch velocity vs GT beats (outputs_prime/)")
print("=" * 70)

PITCH_FILES = {
    "noprime_gt":       ("pitch_noprime_gt.npy",        0),
    "prime400_gt":      ("pitch_prime400_gt.npy",       400),
    "prime400_shuf":    ("pitch_prime400_shuf.npy",     400),
    "beatprime400_gt":  ("pitch_beatprime400_gt.npy",   400),
    "beatprime400_shuf":("pitch_beatprime400_shuf.npy", 400),
}

gen_curves = {k: [] for k in PITCH_FILES}
gen_beat_curves = []

folders = sorted([
    d for d in os.listdir(PRIME_DIR)
    if os.path.isdir(os.path.join(PRIME_DIR, d))
])

for folder in folders:
    uid = folder.split("_")[0]
    beat_arr, _ = load_beat(uid)
    if beat_arr is None:
        print(f"  skip {folder} — no beat")
        continue

    folder_path = os.path.join(PRIME_DIR, folder)

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
        print(f"  skip {folder} — no beat period")
        continue

    gen_beat_curves.append(b_win)

    for key, (fname, prime_len) in PITCH_FILES.items():
        fpath = os.path.join(folder_path, fname)
        if not os.path.exists(fpath):
            continue

        p = np.load(fpath).astype(np.float32)
        if p.ndim > 1:
            p = p.flatten()
        p = p[:SEQ_LEN]
        if len(p) < SEQ_LEN:
            p = np.pad(p, (0, SEQ_LEN - len(p)))

        cents_gen, sil_gen = tokens_to_cents(p)
        vel = cents_to_velocity(cents_gen, sil_gen)

        vel_eval   = vel[prime_len:]
        b_win_eval = b_win[prime_len:]

        corr, lags = normalized_crosscorr(vel_eval, b_win_eval, MAX_LAG)
        gen_curves[key].append(corr)

    print(f"  {folder}: {sum(len(v) for v in gen_curves.values())} curves accumulated")

print(f"\nFolders processed: {len(folders)}")
for key, curves in gen_curves.items():
    if curves:
        mean_curve = np.mean(curves, axis=0)
        peak_val = mean_curve[center - half_bp: center + half_bp + 1].max()
        print(f"  {key:<22}: {len(curves)} windows, peak CC={peak_val:.4f}")




fig, axes = plt.subplots(1, 2, figsize=(14, 5))
lags_ms = lags / FPS * 1000

ax = axes[0]
ax.plot(lags_ms, gt_mean,   label="GT beat", color="steelblue", lw=2)
ax.plot(lags_ms, shuf_mean, label="Shuffled beat", color="orange", lw=2, ls="--")
ax.plot(lags_ms, diff_mean, label="GT − shuffled", color="green", lw=1.5, ls=":")
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
for k in range(-3, 4):
    ax.axvline(k * mean_beat_period / FPS * 1000, color="red", lw=0.6, alpha=0.3, ls=":")
ax.set_title("GT vocal pitch velocity × GT beats\n(real data)")
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
ax.set_title("Generated pitch velocity × GT beats\n(outputs_prime/)")
ax.set_xlabel("Lag (ms)")
ax.legend(fontsize=8)
ax.set_xlim(-MAX_LAG / FPS * 1000, MAX_LAG / FPS * 1000)

plt.suptitle("Full pitch contour cross-correlation with beat signal", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("contour_crosscorr_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved contour_crosscorr_plot.png")



results = {
    "mean_beat_period_frames": float(mean_beat_period),
    "mean_beat_period_ms": float(mean_beat_period / FPS * 1000),
    "n_gt_windows": len(gt_corr_curves),
    "lags_ms": lags_ms.tolist(),
    "gt_data": {
        "gt_beat_mean_corr":   gt_mean.tolist(),
        "shuf_beat_mean_corr": shuf_mean.tolist(),
        "diff_mean_corr":      diff_mean.tolist(),
        "peak_gt":   float(peak_gt),
        "peak_shuf": float(peak_shuf),
        "gap":       float(peak_gt - peak_shuf),
    },
    "generated": {
        key: {
            "n_windows": len(curves),
            "mean_corr": np.mean(curves, axis=0).tolist() if curves else [],
        }
        for key, curves in gen_curves.items()
    }
}

with open("contour_crosscorr_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved contour_crosscorr_results.json")
print("\nDone.")
