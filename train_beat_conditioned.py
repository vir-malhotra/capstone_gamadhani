"""
Full fine-tuning script for beat-conditioned GaMaDHaNi Stage 1 (diffusion pitch model).
Sketch2Sound approach: one linear beat projection layer, all weights fine-tuned.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from gamadhani.src.hmr_dataset import HMRBeatDataset
from gamadhani.src.model_diffusion import UNet
from gamadhani.utils.generate_utils import load_pitch_fns

PITCH_PATH = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-model.ckpt"
QT_PATH    = "/home/vm2426/.cache/huggingface/hub/models--kmaneeshad--GaMaDHaNi/snapshots/8b8e907d9d8a9981bd8a5f8662499281b8afbd95/diffusion_pitch_model-qt.joblib"
CONFIG     = "configs/diffusion_pitch_config.gin"


# ── Model ─────────────────────────────────────────────────────────────────────

class BeatConditionedUNet(nn.Module):
    def __init__(self, pretrained_unet: UNet, beat_dim: int = 1, cfg_prob: float = 0.1):
        super().__init__()
        self.unet = copy.deepcopy(pretrained_unet)
        self.beat_projection = nn.Linear(beat_dim, self.unet.initial_projection.out_channels)
        self.cfg_prob = cfg_prob
        self.unet.inp_dim = 1

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, time, beat):
        x = self.unet.initial_projection(x)

        if beat.ndim == 3:
            beat = beat.transpose(1, 2)
        elif beat.ndim == 2:
            beat = beat.unsqueeze(-1)

        beat = self.beat_projection(beat).transpose(1, 2)
        x = x + beat

        time = self.unet.positional_encoding(time)

        def _cat_time(x_, t_):
            return torch.cat([x_, t_.unsqueeze(2).expand(-1, -1, x_.shape[-1])], dim=-2)

        skips = []
        for dl in self.unet.downsample_layers:
            skips.append(x)
            x = _cat_time(x, time)
            x = dl(x)
        skips.append(x)

        x = x.permute(0, 2, 1)
        x = self.unet.attention_layers(x)
        x = x.permute(0, 2, 1)

        for ul in self.unet.upsample_layers:
            x = _cat_time(x, time)
            x = torch.cat([x, skips.pop(-1)], dim=1)
            x = ul(x)
        x = torch.cat([x, skips.pop(-1)], dim=1)

        return self.unet.final_projection(x)

    def loss(self, x, beat):
        px, pad = self.unet.pad_to(x, self.unet.strides_prod)
        pb, _   = self.unet.pad_to(beat, self.unet.strides_prod)

        if self.training and self.cfg_prob > 0:
            mask = (torch.rand(pb.shape[0], device=pb.device) < self.cfg_prob)
            pb = pb * (~mask).float()[:, None, None]

        t     = torch.rand((px.shape[0],), device=px.device)
        noise = torch.randn_like(px)
        x_t   = t[:, None, None] * px + (1.0 - t[:, None, None]) * noise

        pred   = self.forward(x_t, t, pb)
        target = x - self.unet.unpad(noise, pad)
        return torch.mean((self.unet.unpad(pred, pad) - target) ** 2)

    def sample(self, beat: torch.Tensor, num_steps: int = 100, guidance_scale: float = 1.0):
        self.eval()
        batch_size = beat.shape[0]
        noise = torch.randn(batch_size, self.unet.inp_dim, beat.shape[-1]).to(self.device)
        padded_noise, padding = self.unet.pad_to(noise, self.unet.strides_prod)
        padded_beat, _        = self.unet.pad_to(beat.to(self.device), self.unet.strides_prod)
        null_beat             = torch.zeros_like(padded_beat)
        t_array = torch.ones(batch_size).to(self.device)

        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t, device=self.device)
                pred_cond = self.forward(padded_noise, t_tensor * t_array, padded_beat)
                if guidance_scale != 1.0:
                    pred_null = self.forward(padded_noise, t_tensor * t_array, null_beat)
                    pred = pred_null + guidance_scale * (pred_cond - pred_null)
                else:
                    pred = pred_cond
                padded_noise = padded_noise + (1.0 / num_steps) * pred

        return self.unet.unpad(padded_noise, padding)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="beat-conditioned-gamadhani",
        name=args.run_name,
        config={
            "lr":           args.lr,
            "batch_size":   args.batch_size,
            "epochs":       args.epochs,
            "beat_dim":     args.beat_dim,
            "cfg_prob":     args.cfg_prob,
            "val_ratio":    args.val_ratio,
            "seed":         args.seed,
        },
    )

    # Model
    pitch_model, pitch_qt, pitch_task_fn, invert_pitch_fn, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)

    beat_model = BeatConditionedUNet(
        pitch_model, beat_dim=args.beat_dim, cfg_prob=args.cfg_prob
    ).to(device)

    n_params = sum(p.numel() for p in beat_model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    # Data
    train_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="train",
                               val_ratio=args.val_ratio, seed=args.seed,
                               window_stride=args.window_stride, seq_len=args.seq_len)
    val_ds   = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                               val_ratio=args.val_ratio, seed=args.seed,
                               window_stride=args.window_stride, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    wandb.config.update({"train_windows": len(train_ds), "val_windows": len(val_ds)})

    optimizer = torch.optim.AdamW(beat_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        beat_model.load_state_dict(ckpt["state_dict"])
        if args.fresh_optimizer:
            # Fresh optimizer + scheduler — use this when starting a new training
            # phase (e.g. CFG fine-tune) so the LR schedule begins from the top.
            print(f"Resumed weights from epoch {ckpt['epoch']} with fresh optimizer (lr={args.lr})")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            # fast-forward scheduler to match resumed epoch
            for _ in range(ckpt["epoch"]):
                scheduler.step()
            print(f"Resumed from epoch {ckpt['epoch']}, continuing scheduler")
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = float("inf") if args.reset_best_val else ckpt["val_loss"]
        global_step   = ckpt["epoch"] * len(train_loader)
        print(f"  best val loss starting at: {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ──
        beat_model.train()
        train_loss = 0.0
        for batch in train_loader:
            x    = batch["normalized_pitch"].to(device).unsqueeze(1)
            beat = batch["beat"].to(device).unsqueeze(1)

            loss = beat_model.loss(x, beat)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(beat_model.parameters(), 1.0)
            optimizer.step()

            train_loss  += loss.item()
            global_step += 1
            wandb.log({"train/loss_step": loss.item(), "step": global_step})

        scheduler.step()
        avg_train = train_loss / len(train_loader)

        # ── val ──
        beat_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x    = batch["normalized_pitch"].to(device).unsqueeze(1)
                beat = batch["beat"].to(device).unsqueeze(1)
                val_loss += beat_model.loss(x, beat).item()
        avg_val = val_loss / len(val_loader)

        lr_now = scheduler.get_last_lr()[0]
        wandb.log({
            "train/loss_epoch": avg_train,
            "val/loss":         avg_val,
            "lr":               lr_now,
            "epoch":            epoch,
        })

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  lr={lr_now:.2e}")

        # ── checkpoint ──
        # Only keep 2 files: best.ckpt and latest.ckpt (overwrite each time)
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            ckpt_path = os.path.join(args.ckpt_dir, "best.ckpt")
            torch.save({
                "epoch":       epoch,
                "state_dict":  beat_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    avg_val,
                "train_loss":  avg_train,
                "config":      vars(args),
            }, ckpt_path)
            print(f"  → new best val loss: {best_val_loss:.4f}  saved {ckpt_path}")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.ckpt_dir, "latest.ckpt")
            torch.save({
                "epoch":       epoch,
                "state_dict":  beat_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "val_loss":    avg_val,
                "train_loss":  avg_train,
                "config":      vars(args),
            }, ckpt_path)

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",     default="hmr-gt-beats-1ch")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--beat_dim",     type=int,   default=1)
    parser.add_argument("--cfg_prob",     type=float, default=0.1, help="fraction of training samples where beat is zeroed for CFG")
    parser.add_argument("--val_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--save_every",   type=int,   default=10)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--ckpt_dir",     default="checkpoints/hmr_gt_beats")
    parser.add_argument("--window_stride", type=int,   default=1200, help="stride between windows (1200=non-overlapping, 600=50% overlap)")
    parser.add_argument("--seq_len",       type=int,   default=1200, help="window length in frames (1200=12s, 2000=20s, 3000=30s)")
    parser.add_argument("--resume",         default=None, help="path to checkpoint to resume from")
    parser.add_argument("--reset_best_val", action="store_true", help="reset best_val_loss to inf on resume (use when val set changes)")
    parser.add_argument("--fresh_optimizer", action="store_true", help="on resume, use a fresh optimizer+scheduler instead of restoring from checkpoint (use for new training phases)")
    args = parser.parse_args()
    train(args)
