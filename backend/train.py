"""
BrainVision 3D - Training Pipeline for BraTS 2020 Brain Tumor Segmentation
Hardware Optimized: NVIDIA RTX 3050 (6 GB VRAM)

Design:
- Patch-based training (default patch size: 96x96x96)
- Automatic Mixed Precision (AMP with GradScaler)
- Small batch size (default: 1 or 2) with gradient accumulation
- InstanceNorm3d & memory-efficient residual 3D U-Net
- Checkpoint saving & BraTS validation metrics (WT, TC, ET Dice)
- Synthetic benchmark generation for instant validation
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .models.unet3d import UNet3D, ResidualUNet3D, count_parameters
from .dataset import BraTS2020Dataset, generate_synthetic_case
from .losses import DiceCELoss, compute_brats_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D U-Net on BraTS 2020 (RTX 3050 6GB Optimized)")
    parser.add_argument("--data_dir", type=str, default="data/brats2020_sample",
                        help="Path to directory containing BraTS subject folders")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[96, 96, 96],
                        help="3D patch size (D H W) (default: 96 96 96 for 6GB VRAM)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Mini-batch size (default: 1 or 2 for RTX 3050)")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Gradient accumulation steps for effective batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Initial learning rate for AdamW")
    parser.add_argument("--base_filters", type=int, default=16,
                        help="U-Net base channel multiplier (16 for 6GB VRAM, 32 for >12GB)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--samples_per_volume", type=int, default=4,
                        help="Patches extracted per volume per epoch")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--generate_synthetic", type=int, default=0,
                        help="If > 0, generates N synthetic BraTS subjects for test runs")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: str,
    use_amp: bool,
    grad_accum_steps: int = 2
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    num_batches = len(loader)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['seg'].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss, metrics = criterion(logits, targets)
                scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()
        else:
            logits = model(images)
            loss, metrics = criterion(logits, targets)
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

        total_loss += loss.item()
        total_ce += metrics['ce_loss']
        total_dice += metrics['dice_loss']

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, num_batches)
    avg_ce = total_ce / max(1, num_batches)
    avg_dice = total_dice / max(1, num_batches)

    return {
        'loss': avg_loss,
        'ce_loss': avg_ce,
        'dice_loss': avg_dice
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    wt_scores = []
    tc_scores = []
    et_scores = []

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['seg'].to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss, _ = criterion(logits, targets)
        else:
            logits = model(images)
            loss, _ = criterion(logits, targets)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        # Compute BraTS composite metrics per item in batch
        for i in range(preds.shape[0]):
            metrics = compute_brats_metrics(preds[i], targets[i])
            wt_scores.append(metrics['dice_wt'])
            tc_scores.append(metrics['dice_tc'])
            et_scores.append(metrics['dice_et'])

    num_samples = max(1, len(wt_scores))
    mean_wt = sum(wt_scores) / num_samples
    mean_tc = sum(tc_scores) / num_samples
    mean_et = sum(et_scores) / num_samples
    mean_dice = (mean_wt + mean_tc + mean_et) / 3.0

    return {
        'val_loss': total_loss / max(1, len(loader)),
        'val_dice_wt': mean_wt,
        'val_dice_tc': mean_tc,
        'val_dice_et': mean_et,
        'val_mean_dice': mean_dice
    }


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    use_amp = (not args.no_amp) and (device == "cuda")

    print("=" * 60)
    print("BrainVision 3D - BraTS 2020 Tumor Segmentation Training")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Patch Size: {tuple(args.patch_size)}")
    print(f"Batch Size: {args.batch_size} (Accumulation: {args.grad_accum_steps} steps)")
    print(f"Automatic Mixed Precision (AMP): {use_amp}")
    print(f"Base Channel Filters: {args.base_filters}")
    print("=" * 60)

    # Generate synthetic cases if requested or if data_dir is missing
    if args.generate_synthetic > 0:
        print(f"Generating {args.generate_synthetic} synthetic BraTS volume(s) in {args.data_dir}...")
        for i in range(args.generate_synthetic):
            generate_synthetic_case(args.data_dir, subject_id=f"BraTS20_Synthetic_{i+1:03d}")

    # Initialize Dataset
    patch_size = tuple(args.patch_size)
    dataset = BraTS2020Dataset(
        data_dir=args.data_dir,
        patch_size=patch_size,
        is_training=True,
        samples_per_volume=args.samples_per_volume,
        p_foreground=0.75
    )

    if len(dataset) == 0:
        print(f"Error: No subject data found in {args.data_dir}.")
        print("Tip: Run with '--generate_synthetic 2' to test the full pipeline immediately.")
        sys.exit(1)

    # Train / Val Split (80% / 20%)
    total_samples = len(dataset)
    val_size = max(1, int(0.2 * total_samples))
    train_size = total_samples - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safe for Windows
        pin_memory=(device == "cuda")
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )

    print(f"Training patches: {len(train_set)} | Validation patches: {len(val_set)}")

    # Model, Optimizer, Loss, Scaler
    model = ResidualUNet3D(
        in_channels=4,
        out_channels=4,
        base_filters=args.base_filters,
        dropout_p=0.15
    ).to(device)

    print(f"Model trainable parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = DiceCELoss(dice_weight=1.0, ce_weight=1.0)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = -1.0
    history = []

    print("\nStarting Training Loop...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        ep_start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp, args.grad_accum_steps
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        ep_duration = time.time() - ep_start

        log_entry = {
            'epoch': epoch,
            'duration_sec': round(ep_duration, 1),
            'lr': float(optimizer.param_groups[0]['lr']),
            **train_metrics,
            **val_metrics
        }
        history.append(log_entry)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"WT Dice: {val_metrics['val_dice_wt']:.3f} | "
            f"TC Dice: {val_metrics['val_dice_tc']:.3f} | "
            f"ET Dice: {val_metrics['val_dice_et']:.3f} | "
            f"Mean Dice: {val_metrics['val_mean_dice']:.3f}"
        )

        # Save latest checkpoint
        latest_path = os.path.join(args.save_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mean_dice': val_metrics['val_mean_dice'],
            'base_filters': args.base_filters
        }, latest_path)

        # Save best checkpoint
        if val_metrics['val_mean_dice'] > best_dice:
            best_dice = val_metrics['val_mean_dice']
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_mean_dice': best_dice,
                'base_filters': args.base_filters
            }, best_path)
            print(f"  * New best model saved! (Mean Dice: {best_dice:.3f})")

    # Save training log JSON
    history_path = os.path.join(args.save_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Mean Dice: {best_dice:.3f}")
    print(f"Checkpoints and history saved to '{args.save_dir}/'")
    print("=" * 60)


if __name__ == "__main__":
    main()
