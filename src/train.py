#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
from datetime import timedelta
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from skimage.metrics import structural_similarity as compute_ssim
from torchvision.models import vgg16

from dataset import SRDataset
from model import load_sr_model
from evaluate import evaluate  # for optional final evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler(enabled=self.args.amp)
        self.best_metrics = {"loss": float("inf"), "psnr": 0.0, "ssim": 0.0}
        self.current_epoch = 0

        # Directories
        self.checkpoint_dir = Path(self.args.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tb_log_dir = self.checkpoint_dir.parent / "tensorboard"
        self.tb_log_dir.mkdir(exist_ok=True)

        self.samples_dir = self.checkpoint_dir.parent / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.init_model()
        self.init_datasets()
        self.init_optimizer()
        self.init_loss()
        self.init_tensorboard()
        self.log_config()

    def init_model(self):
        logger.info("🔍 Loading model.")
        self.model = load_sr_model(
            scale=self.args.scale,
            device=self.device,
            use_light_blocks=self.args.light_blocks,
            progressive_scale=self.args.progressive,
            use_compile=self.args.compile
        )
        if self.args.compile:
            try:
                self.model = torch.compile(self.model)
                logger.info("🚀 Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"⚠️ Could not compile model: {e}")

        if torch.cuda.device_count() > 1:
            logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(self.model)

    def init_datasets(self):
        logger.info("📊 Preparing datasets.")
        self.train_ds = SRDataset(
            root_dir=self.args.data_dir,
            split="train",
            scale=self.args.scale,
            patch_size=self.args.patch_size,
            augment=self.args.augment,
            cache=self.args.cache
        )
        self.val_ds = SRDataset(
            root_dir=self.args.data_dir,
            split="val",
            scale=self.args.scale,
            patch_size=None,
            augment=False,
            cache=False
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=self.args.num_workers > 0
        )
        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=min(2, self.args.num_workers),
            pin_memory=True
        )
        logger.info(f"✅ Datasets loaded - Train: {len(self.train_ds)}, Val: {len(self.val_ds)}")

    def init_optimizer(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )

    def init_loss(self):
        self.criterion = torch.nn.L1Loss()
        if self.args.perceptual_weight > 0.0:
            logger.info("🔍 Loading VGG for perceptual loss.")
            vgg = vgg16(pretrained=True).features[:16].to(self.device).eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
            self.perceptual_criterion = torch.nn.L1Loss()

    def init_tensorboard(self):
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)

    def log_config(self):
        logger.info("\n⚙️  Training Configuration:")
        config = {
            "Device": self.device,
            "Scale": f"x{self.args.scale}",
            "Epochs": self.args.epochs,
            "Batch Size": self.args.batch_size,
            "Learning Rate": self.args.lr,
            "Weight Decay": self.args.weight_decay,
            "Patch Size": self.args.patch_size or "Whole Image",
            "Light Blocks": self.args.light_blocks,
            "Progressive Scale": self.args.progressive,
            "AMP": self.args.amp,
            "Compile": self.args.compile,
            "Augment": self.args.augment,
            "Cache": self.args.cache,
            "Perceptual Weight": self.args.perceptual_weight,
            "Model Parameters": f"{sum(p.numel() for p in self.model.parameters()):,}"
        }
        for k, v in config.items():
            logger.info(f"  {k}: {v}")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.train_loader):
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.amp):
                sr_imgs = self.model(lr_imgs)
                loss = self.criterion(sr_imgs, hr_imgs)

                if self.args.perceptual_weight > 0.0:
                    # Convert from [-1..+1] to [0..1] for VGG
                    sr_01 = (sr_imgs + 1.0) / 2.0
                    hr_01 = (hr_imgs + 1.0) / 2.0
                    sr_feats = self.vgg(sr_01)
                    hr_feats = self.vgg(hr_01)
                    perceptual_loss = self.perceptual_criterion(sr_feats, hr_feats)
                    loss = loss + self.args.perceptual_weight * perceptual_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += loss.item()

            if batch_idx % self.args.log_interval == 0:
                batch_time = (time.time() - start_time) / (batch_idx + 1)
                samples_per_sec = self.args.batch_size / batch_time
                logger.debug(
                    f"  Batch {batch_idx}/{len(self.train_loader)} - "
                    f"Loss: {loss.item():.4f} - "
                    f"Speed: {samples_per_sec:.1f} img/sec"
                )

        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/train", avg_loss, self.current_epoch)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        psnr_vals = []
        ssim_vals = []

        for lr_imgs, hr_imgs in self.val_loader:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            with autocast(enabled=self.args.amp):
                sr_raw = self.model(lr_imgs)  # in [-1..+1]
                sr_01 = (sr_raw + 1.0) / 2.0  # now in [0..1]
                hr_01 = (hr_imgs + 1.0) / 2.0  # hr_imgs was [-1..+1], now [0..1]
                val_loss += self.criterion(sr_01, hr_01).item()

            # Convert to uint8 numpy for metrics
            sr_np = (sr_01.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
            hr_np = (hr_01.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)

            psnr_val = 0.0
            ssim_val = 0.0
            try:
                psnr_val = 10 * np.log10(255**2 / np.mean((hr_np - sr_np) ** 2))
            except Exception as e:
                logger.warning(f"PSNR calculation failed: {e}")

            try:
                ssim_val = compute_ssim(hr_np, sr_np, channel_axis=-1, data_range=255)
            except Exception as e:
                logger.warning(f"SSIM calculation failed: {e}")

            psnr_vals.append(psnr_val)
            ssim_vals.append(ssim_val)

        avg_loss = val_loss / len(self.val_loader)
        avg_psnr = sum(psnr_vals) / len(psnr_vals) if len(psnr_vals) > 0 else 0.0
        avg_ssim = sum(ssim_vals) / len(ssim_vals) if len(ssim_vals) > 0 else 0.0

        self.writer.add_scalar("Loss/val", avg_loss, self.current_epoch)
        self.writer.add_scalar("PSNR/val", avg_psnr, self.current_epoch)
        self.writer.add_scalar("SSIM/val", avg_ssim, self.current_epoch)

        return avg_loss, avg_psnr, avg_ssim

    def save_checkpoint(self, is_best: bool = False):
        ckpt_path = self.checkpoint_dir / f"epoch_{self.current_epoch}.pth"
        
        # Get model configuration
        if hasattr(self.model, 'module'):  # DataParallel
            base_model = self.model.module
        else:
            base_model = self.model
        
        model_config = {
            'num_feat': 64,  # You can extract this from base_model if needed
            'num_block': 23,  # You can extract this from base_model if needed
            'num_grow_ch': 32,
            'scale': self.args.scale,
            'use_light_blocks': self.args.light_blocks,
            'progressive_scale': self.args.progressive
        }
        
        state = {
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_metrics": self.best_metrics,
            "model_config": model_config  # Add model configuration
        }
        torch.save(state, str(ckpt_path))
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(state, str(best_path))
            logger.info("💾 Saved new best model.")

    def run(self):
        logger.info("🚀 Starting training...")
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_psnr, val_ssim = self.validate()
            self.scheduler.step(val_loss)

            # Check for new best
            if val_loss < self.best_metrics["loss"]:
                self.best_metrics.update({
                    "loss": val_loss,
                    "psnr": val_psnr,
                    "ssim": val_ssim
                })
                self.save_checkpoint(is_best=True)
            elif epoch % self.args.checkpoint_freq == 0:
                self.save_checkpoint(is_best=False)

            epoch_time = time.time() - epoch_start_time
            elapsed_time = time.time() - start_time
            avg_time_per_epoch = elapsed_time / epoch
            remaining_time = (self.args.epochs - epoch) * avg_time_per_epoch

            logger.info(
                f"\n📊 Epoch {epoch}/{self.args.epochs} | "
                f"Time: {timedelta(seconds=epoch_time)} | "
                f"Remaining: ~{timedelta(seconds=remaining_time)}\n"
                f"  Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}\n"
                f"  PSNR: {val_psnr:.2f} dB | "
                f"SSIM: {val_ssim:.4f} | "
                f"Best PSNR: {self.best_metrics['psnr']:.2f} dB | "
                f"Best SSIM: {self.best_metrics['ssim']:.4f}"
            )

        total_time = time.time() - start_time
        logger.info(f"\n🎉 Training completed in {timedelta(seconds=total_time)}")

        if self.args.final_eval:
            logger.info("\n🔍 Running final evaluation.")
            eval_results = evaluate(
                data_dir=self.args.data_dir,
                model_path=str(self.checkpoint_dir / "best.pth"),
                scale=self.args.scale,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                save_results=True,
                output_dir=str(self.samples_dir),
                use_light_blocks=self.args.light_blocks,
                progressive_scale=self.args.progressive
            )
            logger.info(f"\n📊 Final Evaluation Results:")
            logger.info(f"  PSNR: {eval_results['psnr']:.2f} dB")
            logger.info(f"  SSIM: {eval_results['ssim']:.4f}")

        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Super-Resolution Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory for train/val folders (containing 'train/hr', 'train/lr', 'val/hr', 'val/lr')"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Upscaling factor"
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=128,
        help="Crop patch size for HR during training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay"
    )
    parser.add_argument(
        "--perceptual_weight",
        type=float,
        default=0.1,
        help="Weight for perceptual (VGG) loss"
    )
    parser.add_argument(
        "--light_blocks",
        action="store_true",
        help="Use lightweight RRDB blocks"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive upscaling"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache dataset in RAM"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed‐precision (AMP) training"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile()"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="weights/checkpoints",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="How many batches between logging"
    )
    parser.add_argument(
        "--final_eval",
        action="store_true",
        help="Run a final evaluation at the end of training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()