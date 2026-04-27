"""
HMRBeatDataset — replaces the toy BeatDataset in beat-condition-finetuning.ipynb.

Each item is a 12-second window (1200 frames at 100Hz) containing:
  - "beat"             : [1200]  float32  — beat pulse (ch0 of 3-channel beat signal)
  - "normalized_pitch" : [1200]  float32  — QuantileTransformed pitch (model input)

Exclusions are driven solely by the hmr_instruments.csv (sarangi/violin/excluded flags).
No per-window confidence filtering is applied.
"""

import csv
import os
import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN  = 1200  # frames (12s at 100Hz)
SMOOTH_KERNEL = np.ones(5, dtype=np.float32)  # same kernel used when building beats


def _render_beat_pulses(events: np.ndarray, length: int) -> np.ndarray:
    """Given beat event frame indices, re-render a smoothed pulse signal."""
    sig = np.zeros(length, dtype=np.float32)
    for e in events:
        if 0 <= e < length:
            sig[e] = 1.0
    sig = np.convolve(sig, SMOOTH_KERNEL, mode='same')
    return np.clip(sig, 0.0, 1.0)


def _extract_events(beat_signal: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Cluster non-zero frames into single beat event positions."""
    frames = np.where(beat_signal > threshold)[0]
    if len(frames) == 0:
        return np.array([], dtype=int)
    events, grp = [], [int(frames[0])]
    for f in frames[1:]:
        if f - grp[-1] <= 10:
            grp.append(int(f))
        else:
            events.append(int(np.mean(grp)))
            grp = [int(f)]
    events.append(int(np.mean(grp)))
    return np.array(events, dtype=int)


def augment_beat(
    beat: np.ndarray,
    section_dropout_p: float = 0.3,
    section_dropout_frac: float = 0.3,
    tempo_aug_p: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply beat signal augmentations during training.

    1. Section dropout  — zero out a contiguous random block (length drawn from
       [0.15, section_dropout_frac] × SEQ_LEN).
    2. Tempo halving    — keep every other beat (doubles inter-beat interval).
    3. Tempo doubling   — insert a synthetic beat halfway between each pair
                          (halves inter-beat interval).

    Only one of (2) or (3) is applied per sample, chosen with equal probability.
    Augmentations are independent and can stack (dropout then tempo change).
    """
    if rng is None:
        rng = np.random.default_rng()

    beat = beat.copy()
    T = len(beat)

    # 1. Section dropout
    if rng.random() < section_dropout_p:
        frac   = rng.uniform(0.15, section_dropout_frac)
        length = max(1, int(T * frac))
        start  = int(rng.integers(0, T - length))
        beat[start:start + length] = 0.0

    # 2. Tempo augmentation
    if rng.random() < tempo_aug_p:
        events = _extract_events(beat)
        if len(events) >= 4:
            choice = rng.choice(['half', 'double'])
            if choice == 'half':
                events = events[::2]
            else:  # double
                midpoints = ((events[:-1] + events[1:]) // 2)
                events = np.sort(np.concatenate([events, midpoints]))
            beat = _render_beat_pulses(events, T)

    return beat

PITCH_DIR = "/home/vm2426/HMR_processed/pitch"
BEAT_DIR  = "/home/vm2426/HMR_processed/beats"
HMDF_CSV  = "/home/vm2426/HMDf.xlsx - HMDf.csv"
INSTR_CSV = "/home/vm2426/beat_transformer_hindustani/inference/hmr_instruments.csv"


def _get_valid_uids() -> List[str]:
    """Return UIDs that pass all exclusion criteria and have pitch+beat files on disk."""
    exclude = set()
    with open(INSTR_CSV) as f:
        for row in csv.DictReader(f):
            if (row["sarangi"].strip() == "True"
                    or row["violin"].strip() == "True"
                    or row.get("excluded", "").strip() == "True"):
                exclude.add(row["UID"].strip())

    uids = []
    with open(HMDF_CSV) as f:
        for row in csv.DictReader(f):
            if row["Instrument Code"].strip().upper() != "V":
                continue
            uid = row["UID"].strip()
            if uid in exclude:
                continue
            if not os.path.exists(os.path.join(PITCH_DIR, f"{uid}_f0.npy")):
                continue
            if not os.path.exists(os.path.join(BEAT_DIR, f"{uid}_beat.npy")):
                continue
            uids.append(uid)
    return uids


def _build_windows(uids: List[str], stride: int = SEQ_LEN, seq_len: int = SEQ_LEN):
    """Pre-compute (uid, start_frame, n_frames) windows with given stride.
    stride=seq_len gives non-overlapping windows (default).
    stride<seq_len gives overlapping windows (e.g. stride=600 → 50% overlap).
    """
    windows = []
    for uid in uids:
        f0   = np.load(os.path.join(PITCH_DIR, f"{uid}_f0.npy"))
        beat = np.load(os.path.join(BEAT_DIR,  f"{uid}_beat.npy"))
        n_frames = min(len(f0), beat.shape[1])
        for start in range(0, n_frames - seq_len, stride):
            windows.append((uid, start, n_frames))
    return windows


class HMRBeatDataset(Dataset):
    def __init__(
        self,
        pitch_task_fn,
        pitch_qt,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        beat_augment: bool = True,
        section_dropout_p: float = 0.3,
        section_dropout_frac: float = 0.3,
        tempo_aug_p: float = 0.3,
        condition_on_pitch: bool = False,  # if True, return pitch contour as conditioning instead of beat
        window_stride: int = SEQ_LEN,     # stride between windows; <SEQ_LEN gives overlapping windows
        seq_len: int = SEQ_LEN,           # window length in frames (1200=12s, 2000=20s, 3000=30s)
    ):
        super().__init__()
        self.pitch_task_fn = pitch_task_fn
        self.pitch_qt      = pitch_qt
        self.is_train      = (split == "train")
        self.beat_augment  = beat_augment and self.is_train
        self.section_dropout_p    = section_dropout_p
        self.section_dropout_frac = section_dropout_frac
        self.tempo_aug_p          = tempo_aug_p
        self.condition_on_pitch   = condition_on_pitch
        self.seq_len              = seq_len

        uids = sorted(_get_valid_uids())

        rng = random.Random(seed)
        rng.shuffle(uids)
        n_val = max(1, int(len(uids) * val_ratio))
        val_uids   = set(uids[:n_val])
        train_uids = set(uids[n_val:])
        split_uids = train_uids if split == "train" else val_uids

        stride = window_stride if window_stride != SEQ_LEN else seq_len
        self.windows = _build_windows(list(split_uids), stride=stride, seq_len=seq_len)
        print(f"HMRBeatDataset [{split}]: {len(split_uids)} recordings, "
              f"{len(self.windows)} windows")

        self._f0_cache   = {}
        self._beat_cache = {}

    def _load(self, uid: str):
        if uid not in self._f0_cache:
            self._f0_cache[uid]   = np.load(os.path.join(PITCH_DIR, f"{uid}_f0.npy"))
            self._beat_cache[uid] = np.load(os.path.join(BEAT_DIR,  f"{uid}_beat.npy"))
        return self._f0_cache[uid], self._beat_cache[uid]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        uid, start, n_frames = self.windows[idx]
        f0, beat = self._load(uid)
        end = start + self.seq_len

        f0_window = f0[start:end].astype(np.float32)
        norm_pitch = self.pitch_task_fn(
            inputs={"pitch": {"data": f0_window}},
            time_downsample=1,
            qt_transform=self.pitch_qt,
            add_noise_to_silence=True,
            seq_len=None,
        )["sampled_sequence"]

        # beat[0] = beat pulse, beat[1] = sam pulse, beat[2] = cycle pos
        # 1-channel for now — switch to beat[:, start:end] for 3-channel (set beat_dim=3)
        beat_window = beat[0, start:end].astype(np.float32)  # [SEQ_LEN]

        if self.condition_on_pitch:
            # Use pitch contour values as the conditioning signal instead of beat annotations
            condition = norm_pitch  # [SEQ_LEN]
        else:
            if self.beat_augment:
                beat_window = augment_beat(
                    beat_window,
                    section_dropout_p=self.section_dropout_p,
                    section_dropout_frac=self.section_dropout_frac,
                    tempo_aug_p=self.tempo_aug_p,
                )
            condition = beat_window  # [SEQ_LEN]

        return {
            "beat":             torch.tensor(condition,   dtype=torch.float32),
            "normalized_pitch": torch.tensor(norm_pitch,  dtype=torch.float32),
            "uid":              uid,
            "start":            start,
        }
