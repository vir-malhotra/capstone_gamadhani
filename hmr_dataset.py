"""
HMR dataset for beat-conditioned GaMaDHaNi fine-tuning.

Reads from the LMDB built by build_hmr_lmdb.py.
Each __getitem__ randomly samples a valid 12s window from a recording,
applying per-window CREPE confidence filtering.

Returns:
    normalized_pitch : torch.Tensor [T]         — quantile-transformed pitch tokens
    beat             : torch.Tensor [3, T]       — (beat_pulse, sam_pulse, cycle_pos)
"""

import os
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
from sklearn.preprocessing import QuantileTransformer

from gamadhani.src.protobuf.data_example import AudioExample

SEQ_LEN         = 1200    # 12s at 100Hz — must match gin config
MIN_CONF        = 0.5     # per-frame CREPE confidence threshold
MIN_CONF_FRAC   = 0.5     # drop windows where > 50% of frames are below MIN_CONF
MAX_WINDOW_TRIES = 20     # how many random windows to try before accepting best available


def _normalize_pitch(f0: np.ndarray, pitch_qt: Optional[QuantileTransformer]) -> np.ndarray:
    """
    Replicate pitch_read_downsample_diff normalization exactly (time_downsample=1).
    Input : f0 in Hz, shape [SEQ_LEN], zeros = unvoiced
    Output: normalized tokens, shape [SEQ_LEN]
    """
    MIN_NORM_PITCH = -4915
    PITCH_DOWNSAMPLE = 10
    MIN_CLIP, MAX_CLIP = 200, 600
    SILENCE_TOKEN = MIN_CLIP - 4   # 196

    norm = f0.copy().astype(np.float64)
    norm[norm == 0] = np.nan

    # Hz → cents relative to 440 Hz
    norm[~np.isnan(norm)] = 1200 * np.log2(norm[~np.isnan(norm)] / 440)
    # discretize
    norm[~np.isnan(norm)] = np.around(norm[~np.isnan(norm)])
    # shift
    norm[~np.isnan(norm)] = norm[~np.isnan(norm)] - MIN_NORM_PITCH
    # bin
    norm[~np.isnan(norm)] = norm[~np.isnan(norm)] // PITCH_DOWNSAMPLE + 1
    # clip
    norm[~np.isnan(norm)] = np.clip(norm[~np.isnan(norm)], MIN_CLIP, MAX_CLIP)
    # silence token
    norm[np.isnan(norm)] = SILENCE_TOKEN

    if pitch_qt is not None:
        norm = pitch_qt.transform(norm.reshape(-1, 1)).reshape(-1)

    return norm.astype(np.float32)


class HMRBeatDataset(Dataset):
    def __init__(
        self,
        db_path: str,
        pitch_qt: Optional[QuantileTransformer] = None,
        seq_len: int = SEQ_LEN,
        min_conf: float = MIN_CONF,
        min_conf_frac: float = MIN_CONF_FRAC,
        beat_channels: int = 3,   # 1=pulse only, 3=pulse+sam+cycle_pos
        condition_on_pitch: bool = False,  # if True, use pitch contour as conditioning instead of beat
        repeat_factor: int = 1,   # sample each recording N times per epoch
    ):
        self.db_path            = db_path
        self.pitch_qt           = pitch_qt
        self.seq_len            = seq_len
        self.min_conf           = min_conf
        self.min_conf_frac      = min_conf_frac
        self.beat_channels      = beat_channels
        self.condition_on_pitch = condition_on_pitch
        self.repeat_factor      = repeat_factor
        self._env               = None
        self._keys              = None

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(self.db_path, lock=False, readahead=False)
        return self._env

    @property
    def keys(self):
        if self._keys is None:
            with self.env.begin(write=False) as txn:
                self._keys = list(txn.cursor().iternext(values=False))
        return self._keys

    def __len__(self):
        return len(self.keys) * self.repeat_factor

    def __getitem__(self, index):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index % len(self.keys)]))

        d    = ae.as_dict()
        f0   = d["pitch"]["data"].astype(np.float32)          # [N]
        beat = d["beat"]["data"].astype(np.float32)            # [3, N]
        conf = d["beat_confidence"]["data"].astype(np.float32) # [N]

        n = min(f0.shape[0], beat.shape[1], conf.shape[0])
        f0, beat, conf = f0[:n], beat[:, :n], conf[:n]

        max_start = n - self.seq_len
        if max_start <= 0:
            start = 0
        else:
            # try to find a window with acceptable confidence
            best_start, best_frac = 0, 1.0
            for _ in range(MAX_WINDOW_TRIES):
                s = np.random.randint(0, max_start)
                lo_frac = float((conf[s:s + self.seq_len] < self.min_conf).mean())
                if lo_frac < self.min_conf_frac:
                    start = s
                    break
                if lo_frac < best_frac:
                    best_frac, best_start = lo_frac, s
            else:
                start = best_start  # fall back to least-bad window

        f0_win   = f0[start : start + self.seq_len]
        beat_win = beat[:, start : start + self.seq_len]

        norm_pitch = _normalize_pitch(f0_win, self.pitch_qt)

        if self.condition_on_pitch:
            # Use the normalized pitch contour itself as the conditioning signal [1, T]
            condition = torch.tensor(norm_pitch, dtype=torch.float32).unsqueeze(0)
        else:
            if self.beat_channels == 1:
                beat_win = beat_win[0:1]  # pulse channel only → [1, T]
            condition = torch.tensor(beat_win, dtype=torch.float32)  # [C, T]

        return {
            "normalized_pitch": torch.tensor(norm_pitch, dtype=torch.float32),  # [T]
            "beat":             condition,  # [C, T] — beat signal or pitch contour
        }
