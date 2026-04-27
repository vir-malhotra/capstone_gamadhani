"""
Cross-correlation based beat alignment evaluation.

Instead of binary F1, compute the normalized cross-correlation between:
  - onset_signal: binary array with 1 at each detected pitch onset
  - beat_signal:  binary array with 1 at each GT beat position

The peak cross-correlation value (max over lags ±50 frames = ±500ms) measures
how well the onset rhythm matches the beat rhythm regardless of phase offset.
The lag at peak tells us if the model is systematically ahead/behind the beat.

Also computes:
  - IOI consistency: std of inter-onset intervals normalized by beat period
  - Beat period accuracy: ratio of modal IOI to GT beat period

Runs on all generated .npy files in outputs/ for both models.
"""

import os, sys, json, copy, random
import numpy as np
import joblib
import torch
import torch.nn as nn
from scipy import signal as scipy_signal
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns
from gamadhani.src.hmr_dataset import HMRBeatDataset
import lmdb
from gamadhani.src.protobuf.data_example import AudioExample

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG     = "configs/diffusion_pitch_config.gin"
GT_CKPT    = "checkpoints/hmr_gt_beats/best.ckpt"
EXT_CKPT   = "checkpoints/hmr_extracted_beats/best.ckpt"
GT_LMDB    = "/home/vm2426/HMR_processed/lmdb/val"
EXT_LMDB   = "/home/vm2426/HMR_processed/lmdb_extracted/val"

NUM_STEPS    = 50
NUM_SAMPLES  = 4
SEQ_LEN      = 1200
ONSET_THR    = 10
MAX_LAG      = 50    # ±50 frames = ±500ms for cross-corr
FPS          = 100


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


def onset_signal(onsets, length):
    sig = np.zeros(length, dtype=np.float32)
    sig[onsets[onsets < length]] = 1.0
    return sig


def beat_signal_from_frames(beat_frames, length):
    sig = np.zeros(length, dtype=np.float32)
    sig[beat_frames[beat_frames < length]] = 1.0
    return sig


def peak_crosscorr(onset_sig, beat_sig, max_lag=MAX_LAG):
    """Normalized cross-correlation peak within ±max_lag frames."""
    if onset_sig.sum() == 0 or beat_sig.sum() == 0:
        return 0.0, 0
    # normalize
    o = onset_sig / (onset_sig.std() + 1e-8)
    b = beat_sig  / (beat_sig.std()  + 1e-8)
    corr = np.correlate(o, b, mode='full')
    corr /= len(onset_sig)
    center = len(corr) // 2
    window = corr[center - max_lag : center + max_lag + 1]
    peak_idx = np.argmax(window)
    return float(window[peak_idx]), peak_idx - max_lag


def ioi_consistency(onsets, gt_beat_period):
    """Normalized IOI std: 0=perfectly regular, 1=random."""
    if len(onsets) < 3 or gt_beat_period <= 0:
        return np.nan
    iois = np.diff(onsets) / FPS   # in seconds
    return float(iois.std() / gt_beat_period)


def beat_period_accuracy(onsets, gt_beat_period):
    """Modal IOI / GT beat period (1.0 = perfect match)."""
    if len(onsets) < 3 or gt_beat_period <= 0:
        return np.nan
    iois = np.diff(onsets).astype(float) / FPS
    # bin IOIs and find modal bin
    hist, edges = np.histogram(iois, bins=20, range=(0, 5))
    modal_ioi = (edges[np.argmax(hist)] + edges[np.argmax(hist)+1]) / 2
    return float(modal_ioi / gt_beat_period)


# ── Load GT beat frames from LMDB ─────────────────────────────────────────────

def load_gt_beats(lmdb_path):
    """Returns {uid: full_pulse_array} from LMDB."""
    result = {}
    env = lmdb.open(lmdb_path, lock=False, readahead=False)
    with env.begin() as txn:
        for key in txn.cursor().iternext(values=False):
            uid = key.decode("ascii")
            ae  = AudioExample(txn.get(key))
            result[uid] = ae.as_dict()["beat"]["data"].astype(np.float32)[0]
    env.close()
    return result


# ── Evaluate one model ────────────────────────────────────────────────────────

def evaluate_model(beat_model, val_lmdb, gt_beat_map, qt, device, label):
    env = lmdb.open(val_lmdb, lock=False, readahead=False)
    windows = []
    with env.begin() as txn:
        for key in txn.cursor().iternext(values=False):
            uid = key.decode("ascii")
            ae  = AudioExample(txn.get(key))
            d   = ae.as_dict()
            beat_full = d["beat"]["data"].astype(np.float32)[0]  # pulse ch
            n = beat_full.shape[0]
            for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                windows.append({"uid": uid, "start": start,
                                 "beat": beat_full[start:start+SEQ_LEN]})
    env.close()

    results = defaultdict(list)

    for i, w in enumerate(windows):
        uid, start = w["uid"], w["start"]
        beat_arr = w["beat"]

        # GT beat frames (from GT annotations)
        gt_pulse = gt_beat_map.get(uid)
        if gt_pulse is None: continue
        gt_win = gt_pulse[start:start+SEQ_LEN]
        gt_beats = np.where(np.diff(np.concatenate([[0],(gt_win>0.5).astype(int)]))==1)[0]
        if len(gt_beats) < 3: continue

        gt_period = np.median(np.diff(gt_beats)) / FPS   # seconds

        beat_in   = torch.tensor(beat_arr).unsqueeze(0).unsqueeze(0).to(device)
        # shuffled: pick different uid
        shuf_candidates = [ww for ww in windows if ww["uid"] != uid]
        shuf_beat = torch.tensor(random.choice(shuf_candidates)["beat"]).unsqueeze(0).unsqueeze(0).to(device)
        beat_zero = torch.zeros_like(beat_in)

        beat_sig = beat_signal_from_frames(gt_beats, SEQ_LEN)

        for _ in range(NUM_SAMPLES):
            for cond_name, cond_beat in [("gt", beat_in), ("shuf", shuf_beat), ("zero", beat_zero)]:
                gen  = beat_model.sample(cond_beat).squeeze().cpu().numpy()
                onsets = detect_onsets(gen, qt)
                o_sig  = onset_signal(onsets, SEQ_LEN)

                peak_cc, lag = peak_crosscorr(o_sig, beat_sig)
                ioi_cons     = ioi_consistency(onsets, gt_period)
                bp_acc       = beat_period_accuracy(onsets, gt_period)

                results[cond_name].append({
                    "peak_cc": peak_cc, "lag": lag,
                    "ioi_cons": ioi_cons, "bp_acc": bp_acc,
                    "n_onsets": len(onsets), "n_beats": len(gt_beats)
                })

        print(f"  [{i+1}/{len(windows)}]", end="\r", flush=True)

    print(f"\n\n{'='*65}")
    print(f"MODEL: {label}")
    print(f"{'='*65}")
    print(f"{'Condition':<10} {'CrossCorr':>10} {'IOI cons↓':>10} {'BP acc':>8} {'Lag(fr)':>8}")
    print(f"{'-'*50}")
    for cond in ["gt", "shuf", "zero"]:
        rows = results[cond]
        cc   = [r["peak_cc"]  for r in rows]
        ic   = [r["ioi_cons"] for r in rows if not np.isnan(r["ioi_cons"])]
        bp   = [r["bp_acc"]   for r in rows if not np.isnan(r["bp_acc"])]
        lags = [r["lag"]      for r in rows]
        print(f"  {cond:<8}   {np.mean(cc):>9.4f}  {np.mean(ic):>9.4f}  {np.mean(bp):>7.4f}  {np.mean(lags):>7.1f}")
    print(f"  GT−shuf gap (CrossCorr): {np.mean([r['peak_cc'] for r in results['gt']]) - np.mean([r['peak_cc'] for r in results['shuf']]):.4f}")
    print(f"{'='*65}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qt = joblib.load(QT_PATH)
    pitch_model, _, _, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)

    gt_beat_map = load_gt_beats(GT_LMDB)

    all_results = {}

    for label, ckpt_path, val_lmdb in [
        ("GT beats model",        GT_CKPT,  GT_LMDB),
        ("Extracted beats model", EXT_CKPT, EXT_LMDB),
    ]:
        model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"\nLoaded {label}: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

        results = evaluate_model(model, val_lmdb, gt_beat_map, qt, device, label)
        all_results[label] = results

    with open("crosscorr_eval_results.json", "w") as f:
        def clean(obj):
            if isinstance(obj, dict): return {k: clean(v) for k,v in obj.items()}
            if isinstance(obj, list): return [clean(v) for v in obj]
            if isinstance(obj, (np.floating, float)): return float(obj) if not np.isnan(obj) else None
            if isinstance(obj, np.integer): return int(obj)
            return obj
        json.dump(clean(all_results), f, indent=2)
    print("\nSaved crosscorr_eval_results.json")


if __name__ == "__main__":
    main()
