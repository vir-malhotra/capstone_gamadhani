"""
Fine-tuning GaMaDHaNi conditioned on the pitch contour itself.

Instead of the 3-channel beat annotation signal, the conditioning input is the
normalized pitch token sequence (QT-transformed, shape [1, T]).

At training time: condition on the GT pitch contour from the same window.
At inference time: provide any pitch contour (from real audio or prior generation)
to guide beat-aligned generation.

Architecture: identical to BeatConditionedUNet but beat_dim=1 (pitch conditioning).
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


# ── Model (identical to BeatConditionedUNet, beat_dim=1) ──────────────────────

class PitchConditionedUNet(nn.Module):
    def __init__(self, pretrained_unet: UNet, pitch_dropout: float = 0.1):
        super().__init__()
        self.unet = copy.deepcopy(pretrained_unet)
        # Project 1-channel pitch condition into model feature space
        self.pitch_projection = nn.Linear(1, self.unet.initial_projection.out_channels)
        self.pitch_dropout = nn.Dropout(pitch_dropout)
        self.unet.inp_dim = 1

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, time, pitch_cond, drop=True):
        x = self.unet.initial_projection(x)

        # pitch_cond: [B, 1, T] → transpose to [B, T, 1] → project → [B, T, C] → transpose back
        pc = pitch_cond.transpose(1, 2)          # [B, T, 1]
        pc = self.pitch_projection(pc)            # [B, T, C]
        pc = pc.transpose(1, 2)                  # [B, C, T]
        if drop:
            pc = self.pitch_dropout(pc)
        x = x + pc

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

    def loss(self, x, pitch_cond):
        px, pad = self.unet.pad_to(x, self.unet.strides_prod)
        pc, _   = self.unet.pad_to(pitch_cond, self.unet.strides_prod)

        t     = torch.rand((px.shape[0],), device=px.device)
        noise = torch.randn_like(px)
        x_t   = t[:, None, None] * px + (1.0 - t[:, None, None]) * noise

        pred   = self.forward(x_t, t, pc, drop=self.training)
        target = x - self.unet.unpad(noise, pad)
        return torch.mean((self.unet.unpad(pred, pad) - target) ** 2)

    def sample(self, pitch_cond: torch.Tensor, num_steps: int = 100):
        self.eval()
        batch_size = pitch_cond.shape[0]
        noise = torch.randn(batch_size, self.unet.inp_dim, self.unet.seq_len).to(self.device)
        padded_noise, padding = self.unet.pad_to(noise, self.unet.strides_prod)
        padded_pc, _          = self.unet.pad_to(pitch_cond.to(self.device), self.unet.strides_prod)
        t_array = torch.ones(batch_size).to(self.device)

        with torch.no_grad():
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t, device=self.device)
                pred = self.forward(padded_noise, t_tensor * t_array, padded_pc, drop=False)
                padded_noise = padded_noise + (1.0 / num_steps) * pred

        return self.unet.unpad(padded_noise, padding)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    import joblib
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(
        project="beat-conditioned-gamadhani",
        name=args.run_name,
        config={
            "lr":              args.lr,
            "batch_size":      args.batch_size,
            "epochs":          args.epochs,
            "pitch_dropout":   args.pitch_dropout,
            "conditioning":    "pitch_contour",
            "seed":            args.seed,
        },
    )

    # Load pretrained model + QT
    pitch_model, pitch_qt, pitch_task_fn, _, _ = load_pitch_fns(
        pitch_path=PITCH_PATH, model_type="diffusion", qt_path=QT_PATH,
        config_path=CONFIG, device=device)

    model = PitchConditionedUNet(
        pitch_model, pitch_dropout=args.pitch_dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    wandb.config.update({"trainable_params": n_params})

    # Dataset: condition_on_pitch=True → returns normalized pitch as the "beat" field
    train_ds = HMRBeatDataset(pitch_task_fn, pitch_qt, split="train",
                               val_ratio=args.val_ratio, seed=args.seed,
                               beat_augment=False, condition_on_pitch=True)
    val_ds   = HMRBeatDataset(pitch_task_fn, pitch_qt, split="val",
                               val_ratio=args.val_ratio, seed=args.seed,
                               beat_augment=False, condition_on_pitch=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    wandb.config.update({"train_windows": len(train_ds), "val_windows": len(val_ds)})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_val_loss = float("inf")
    global_step   = 0
    start_epoch   = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        global_step   = ckpt["epoch"] * len(train_loader)
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']}, best val loss {best_val_loss:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x          = batch["normalized_pitch"].to(device).unsqueeze(1)  # [B, 1, T]
            pitch_cond = batch["beat"].to(device)                           # [B, 1, T]

            loss = model.loss(x, pitch_cond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss  += loss.item()
            global_step += 1
            wandb.log({"train/loss_step": loss.item(), "step": global_step})

        scheduler.step()
        avg_train = train_loss / len(train_loader)

        # ── val ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x          = batch["normalized_pitch"].to(device).unsqueeze(1)
                pitch_cond = batch["beat"].to(device)
                val_loss  += model.loss(x, pitch_cond).item()
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

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   avg_val,
                "train_loss": avg_train,
                "config":     vars(args),
            }, os.path.join(args.ckpt_dir, "best.ckpt"))
            print(f"  → new best: {best_val_loss:.4f}")

        if epoch % args.save_every == 0:
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_loss":   avg_val,
                "train_loss": avg_train,
                "config":     vars(args),
            }, os.path.join(args.ckpt_dir, "latest.ckpt"))

    wandb.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name",      default="hmr-pitch-conditioned")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--pitch_dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio",     type=float, default=0.1)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--save_every",    type=int,   default=10)
    parser.add_argument("--num_workers",   type=int,   default=4)
    parser.add_argument("--ckpt_dir",      default="checkpoints/hmr_pitch_conditioned")
    parser.add_argument("--resume",        default=None)
    args = parser.parse_args()
    train(args)
