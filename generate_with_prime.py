import copy, os, sys, json, random
import numpy as np
import joblib
import lmdb
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.utils.generate_utils import load_pitch_fns
from gamadhani.src.protobuf.data_example import AudioExample
from hmr_dataset import _normalize_pitch

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG     = "configs/diffusion_pitch_config.gin"
BEST_CKPT  = "checkpoints/hmr_gt_beats/best.ckpt"
LMDB_VAL   = "/home/vm2426/HMR_processed/lmdb/val"
OUT_DIR    = "outputs_prime"

SEQ_LEN    = 1200
NUM_STEPS  = 50               
PRIME_LENS = [100, 200, 400]  



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

    def sample(self, beat: torch.Tensor, num_steps: int = NUM_STEPS,
               prime: torch.Tensor = None):
        b = beat.shape[0]
        noise = torch.randn(b, self.unet.inp_dim, self.unet.seq_len).to(self.device)
        pn, pad = self.unet.pad_to(noise, self.unet.strides_prod)
        pb, _   = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        if prime is not None:
            prime = prime.to(self.device)
        t_arr = torch.ones(b).to(self.device)
        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                tt = torch.tensor(t, device=self.device)
                alpha_t = (tt * t_arr).unsqueeze(1).unsqueeze(2)
                if prime is not None:
                    P = prime.shape[-1]
                    pn[:, :, :P] = (1 - alpha_t) * noise[:, :, :P] + alpha_t * prime
                pn = pn + (1.0 / num_steps) * self.forward(pn, tt * t_arr, pb, drop=False)
        return self.unet.unpad(pn, pad)



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    pitch_model, pitch_qt, pitch_task_fn, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)
    qt = joblib.load(QT_PATH)

    beat_model = BeatConditionedUNet(pitch_model, beat_dim=1).to(device)
    ckpt = torch.load(BEST_CKPT, map_location=device)
    beat_model.load_state_dict(ckpt["state_dict"])
    beat_model.eval()
    print(f"Loaded: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    env = lmdb.open(LMDB_VAL, lock=False, readahead=False)
    windows = []
    with env.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        for key in keys:
            ae = AudioExample(txn.get(key))
            d  = ae.as_dict()
            uid  = key.decode("ascii")
            f0   = d["pitch"]["data"].astype(np.float32)
            beat = d["beat"]["data"].astype(np.float32)   # [3, N]
            n    = min(f0.shape[0], beat.shape[1])
            f0, beat = f0[:n], beat[:, :n]
            for start in range(0, n - SEQ_LEN + 1, SEQ_LEN):
                norm = _normalize_pitch(f0[start:start+SEQ_LEN], qt)
                b    = beat[0, start:start+SEQ_LEN]           # pulse channel [T]
                windows.append({"uid": uid, "start": start, "pitch": norm, "beat": b})
    env.close()
    print(f"Val windows: {len(windows)}")

    # For shuffled beat: pick from a different recording
    def get_shuf_beat(i):
        uid_i = windows[i]["uid"]
        candidates = [j for j, w in enumerate(windows) if w["uid"] != uid_i]
        return windows[random.choice(candidates)]["beat"]

    os.makedirs(OUT_DIR, exist_ok=True)

    for i, item in enumerate(windows):
        uid      = item["uid"]
        start    = item["start"]
        pitch_gt = item["pitch"]  
        beat_gt  = item["beat"]

        folder = os.path.join(OUT_DIR, f"{uid}_{start}")
        os.makedirs(folder, exist_ok=True)

        np.save(f"{folder}/pitch_gt.npy", pitch_gt)

        beat_in  = torch.tensor(beat_gt).unsqueeze(0).unsqueeze(0).to(device)   # [1,1,T]
        beat_shuf_np = get_shuf_beat(i)
        beat_shuf = torch.tensor(beat_shuf_np).unsqueeze(0).unsqueeze(0).to(device)

        # No prime, GT beat (baseline)
        gen = beat_model.sample(beat_in).squeeze().cpu().numpy()
        np.save(f"{folder}/pitch_noprime_gt.npy", gen)

        # Pitch prime: GT pitch for first P frames
        for P in PRIME_LENS:
            prime_pitch = torch.tensor(pitch_gt[:P], dtype=torch.float32).reshape(1, 1, P)
            gen_gt   = beat_model.sample(beat_in,   prime=prime_pitch).squeeze().cpu().numpy()
            gen_shuf = beat_model.sample(beat_shuf, prime=prime_pitch).squeeze().cpu().numpy()
            np.save(f"{folder}/pitch_prime{P}_gt.npy",   gen_gt)
            np.save(f"{folder}/pitch_prime{P}_shuf.npy", gen_shuf)

        SILENCE_QT = -1.077
        NOTE_QT    = 0.2
        NOTE_DUR   = 10   
        beat_rises = np.where(np.diff(np.concatenate([[0], (beat_gt > 0.5).astype(int)])) == 1)[0]

        for P in PRIME_LENS:
            synth = np.full(P, SILENCE_QT, dtype=np.float32)
            for bf in beat_rises[beat_rises < P]:
                synth[bf : bf + NOTE_DUR] = NOTE_QT
            prime_beat = torch.tensor(synth).reshape(1, 1, P)
            gen_gt   = beat_model.sample(beat_in,   prime=prime_beat).squeeze().cpu().numpy()
            gen_shuf = beat_model.sample(beat_shuf, prime=prime_beat).squeeze().cpu().numpy()
            np.save(f"{folder}/pitch_beatprime{P}_gt.npy",   gen_gt)
            np.save(f"{folder}/pitch_beatprime{P}_shuf.npy", gen_shuf)

        print(f"  [{i+1}/{len(windows)}] {uid}_{start}  primes={PRIME_LENS}", flush=True)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
