

import os
import csv
import lmdb
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.src.protobuf.data_example import AudioExample

# ── paths ────────────────────────────────────────────────────────────────────
PITCH_DIR  = "/home/vm2426/HMR_processed/pitch"
BEAT_DIR   = "/home/vm2426/HMR_processed/beats_extracted"
INSTR_CSV  = "/home/vm2426/beat_transformer_hindustani/inference/hmr_instruments.csv"
HMDF_CSV   = "/home/vm2426/HMDf.xlsx - HMDf.csv"
OUTPUT_DIR = "/home/vm2426/HMR_processed/lmdb_extracted"

TAAL_TO_ID = {"teentaal": 0, "ektaal": 1, "jhaptaal": 2, "rupak": 3}
TRAIN_RATIO = 0.9
LMDB_MAP_SIZE = 10 * 1024**3  # 10 GB

# ── helpers ───────────────────────────────────────────────────────────────────

def load_excluded_uids():
    excluded = set()
    with open(INSTR_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("excluded", "").strip() == "True":
                excluded.add(row["UID"].strip())
            if row.get("sarangi", "").strip() == "True" or row.get("violin", "").strip() == "True":
                excluded.add(row["UID"].strip())
    return excluded


def load_recordings():
    excluded = load_excluded_uids()
    recordings = []
    with open(HMDF_CSV) as f:
        for row in csv.DictReader(f):
            if row["Instrument Code"].strip().upper() != "V":
                continue
            uid = row["UID"].strip()
            if uid in excluded:
                continue
            # check all 3 files exist
            f0_path   = os.path.join(PITCH_DIR, f"{uid}_f0.npy")
            conf_path = os.path.join(PITCH_DIR, f"{uid}_confidence.npy")
            beat_path = os.path.join(BEAT_DIR,  f"{uid}_beat.npy")
            if not all(os.path.exists(p) for p in [f0_path, conf_path, beat_path]):
                print(f"  WARNING: missing files for UID {uid}, skipping")
                continue
            recordings.append({
                "uid":   uid,
                "taal":  row["Taal"].strip().lower(),
                "laya":  row["Corrected Lay Label"].strip(),
                "f0":    f0_path,
                "conf":  conf_path,
                "beat":  beat_path,
            })
    return recordings


def stratified_split(recordings, train_ratio=TRAIN_RATIO, seed=42):
    """Split recordings 90/10 stratified by taal."""
    rng = np.random.default_rng(seed)
    by_taal = {}
    for r in recordings:
        by_taal.setdefault(r["taal"], []).append(r)

    train, val = [], []
    for taal, recs in by_taal.items():
        rng.shuffle(recs)
        n_train = max(1, round(len(recs) * train_ratio))
        train.extend(recs[:n_train])
        val.extend(recs[n_train:])
        print(f"  {taal}: {n_train} train / {len(recs)-n_train} val")

    return train, val


def write_split(split_name, recordings, output_dir):
    db_path = os.path.join(output_dir, split_name)
    os.makedirs(db_path, exist_ok=True)
    env = lmdb.open(db_path, map_size=LMDB_MAP_SIZE)

    success, errors = 0, 0
    with env.begin(write=True) as txn:
        for idx, rec in enumerate(tqdm(recordings, desc=split_name)):
            try:
                f0   = np.load(rec["f0"]).astype(np.float32)    # [N]
                conf = np.load(rec["conf"]).astype(np.float32)   # [N]
                beat = np.load(rec["beat"]).astype(np.float32)   # [3, N]

                # trim to same length (CREPE vs beat builder may differ by 1-2)
                n = min(f0.shape[0], beat.shape[1])
                f0   = f0[:n]
                conf = conf[:n]
                beat = beat[:, :n]

                ae = AudioExample()

                # pitch buffer
                ae.put(
                    arrays={"pitch": f0},
                    dtype=np.float32,
                    sample_rate=100,
                    data_path=rec["f0"],
                    start_time=0.0,
                )
                # beat buffer (3 channels stored as 2D array)
                ae.put(
                    arrays={"beat": beat},
                    dtype=np.float32,
                    sample_rate=100,
                    data_path=rec["beat"],
                    start_time=0.0,
                )
                # confidence buffer (used for per-window filtering in dataset)
                ae.put(
                    arrays={"beat_confidence": conf},
                    dtype=np.float32,
                    sample_rate=100,
                    data_path=rec["conf"],
                    start_time=0.0,
                )

                # global conditions: taal_id in singer field
                ae.ae.global_conditions.singer = TAAL_TO_ID.get(rec["taal"], -1)
                ae.ae.global_conditions.tonic  = 0.0
                ae.ae.global_conditions.raga   = 0

                key = f"{rec['uid']}".encode("ascii")
                txn.put(key, bytes(ae))
                success += 1

            except Exception as e:
                print(f"\n  ERROR UID={rec['uid']}: {e}")
                errors += 1

    env.close()
    print(f"{split_name}: {success} written, {errors} errors → {db_path}")
    return success


def verify(output_dir):
    print("\n── Verification ──────────────────────────────────────────")
    for split in ["train", "val"]:
        db_path = os.path.join(output_dir, split)
        if not os.path.exists(db_path):
            continue
        env = lmdb.open(db_path, readonly=True, lock=False)
        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            print(f"\n{split}: {len(keys)} entries")
            ae = AudioExample(txn.get(keys[0]))
            d  = ae.as_dict()
            for k in ["pitch", "beat", "beat_confidence"]:
                if k in d:
                    print(f"  {k}: shape={d[k]['data'].shape}, dtype={d[k]['data'].dtype}")
            print(f"  taal_id (singer): {d['global_conditions']['singer']}")
        env.close()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading recordings...")
    recordings = load_recordings()
    print(f"Valid recordings: {len(recordings)}")

    print("\nSplitting by taal:")
    train, val = stratified_split(recordings)
    print(f"Total — train: {len(train)}, val: {len(val)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_split("train", train, OUTPUT_DIR)
    write_split("val",   val,   OUTPUT_DIR)

    verify(OUTPUT_DIR)
    print("\nDone. LMDB at:", OUTPUT_DIR)
