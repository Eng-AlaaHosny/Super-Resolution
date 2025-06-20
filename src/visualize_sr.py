#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from dataset import SRDataset
from model import load_sr_model

def calculate_metrics(hr_img, sr_img):
    """Compute PSNR and SSIM between two uint8 images."""
    hr_arr = hr_img if isinstance(hr_img, np.ndarray) else np.array(hr_img)
    sr_arr = sr_img if isinstance(sr_img, np.ndarray) else np.array(sr_img)

    hr_gray = hr_arr if hr_arr.ndim == 2 else np.mean(hr_arr, axis=2)
    sr_gray = sr_arr if sr_arr.ndim == 2 else np.mean(sr_arr, axis=2)

    psnr_val = psnr_fn(hr_gray, sr_gray, data_range=255)
    ssim_val = ssim_fn(hr_gray, sr_gray, data_range=255)
    return psnr_val, ssim_val

def plot_metrics(metrics_list, model_name, split, scale, out_dir):
    """Plot PSNR/SSIM curves."""
    psnr_vals = [m['psnr'] for m in metrics_list]
    ssim_vals = [m['ssim'] for m in metrics_list]
    samples = list(range(1, len(psnr_vals) + 1))

    paths = {}
    # PSNR plot
    plt.figure()
    plt.plot(samples, psnr_vals, label="PSNR")
    plt.xlabel("Sample Index")
    plt.ylabel("PSNR (dB)")
    plt.title(f"PSNR vs. Sample ({model_name}, {split}, x{scale})")
    plt.legend()
    p1 = out_dir / f"{model_name}_{split}_{scale}x_psnr.png"
    plt.savefig(p1)
    plt.close()
    paths['psnr_plot'] = str(p1)

    # SSIM plot
    plt.figure()
    plt.plot(samples, ssim_vals, label="SSIM")
    plt.xlabel("Sample Index")
    plt.ylabel("SSIM")
    plt.title(f"SSIM vs. Sample ({model_name}, {split}, x{scale})")
    plt.legend()
    p2 = out_dir / f"{model_name}_{split}_{scale}x_ssim.png"
    plt.savefig(p2)
    plt.close()
    paths['ssim_plot'] = str(p2)

    return paths

def visualize(
    data_dir: str,
    model_path: str,
    split: str = 'val',
    scale: int = 4,
    num_samples: int = 5,
    show_diff: bool = False,
    show_table: bool = False,
    verbose: bool = False,
    show_graphs: bool = True,
    light_blocks: bool = False
):
    """Visualize SR results side-by-side and generate metrics."""
    logging.basicConfig(level=logging.INFO)
    out = Path("visualization_outputs")
    out.mkdir(exist_ok=True)

    # Load dataset
    dataset = SRDataset(str(data_dir), split, scale)
    total = len(dataset)
    if num_samples > total:
        logging.warning(f"Requested {num_samples} samples but only {total} available. Using {total}.")
        num_samples = total

    # Use batch_size=1 to avoid variable-size stacking errors
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load model (adjusted for correct signature)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_sr_model(
        scale=scale,
        model_path=model_path,
        device=device,
        use_light_blocks=light_blocks,
        progressive_scale=False
    )
    model.eval()
    device = next(model.parameters()).device

    cols = 3 if show_diff else 2
    rows = num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_samples == 1:
        axes = np.array([axes])
    fig.suptitle(f"SR Comparison: split={split}, scale={scale}x\nModel: {Path(model_path).name}", fontsize=16)

    metrics_list = []
    count = 0

    for idx, (lr_img, hr_img) in enumerate(loader):
        if count >= num_samples:
            break

        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        with torch.no_grad():
            sr_img = model(lr_img)

        # Convert tensors from [-1..+1] → [0..1], then to uint8
        lr_01 = (lr_img[0] + 1.0) / 2.0
        lr_pil = (lr_01.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        sr_01 = (sr_img[0] + 1.0) / 2.0
        sr_pil = (sr_01.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        hr_01 = (hr_img[0] + 1.0) / 2.0
        hr_pil = (hr_01.cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

        name = f"sample_{count}"

        row = count
        if cols == 3:
            axes[row, 0].imshow(lr_pil)
            axes[row, 0].set_title(f"LR: {name}")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(sr_pil)
            axes[row, 1].set_title(f"SR: {name}")
            axes[row, 1].axis('off')

            diff = np.abs(hr_pil.astype(int) - sr_pil.astype(int)).astype(np.uint8)
            im = axes[row, 2].imshow(diff, cmap='jet')
            axes[row, 2].set_title("Diff Map")
            axes[row, 2].axis('off')
            fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        else:
            axes[row, 0].imshow(lr_pil)
            axes[row, 0].set_title(f"LR: {name}")
            axes[row, 0].axis('off')

            axes[row, 1].imshow(sr_pil)
            axes[row, 1].set_title(f"SR: {name}")
            axes[row, 1].axis('off')

        # Calculate metrics on [0..255] uint8
        psnr_val, ssim_val = calculate_metrics(hr_pil, sr_pil)
        metrics_list.append({
            'filename': name,
            'psnr': float(psnr_val),
            'ssim': float(ssim_val)
        })
        logging.debug(f"Processed {name} - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")

        count += 1

    # Save the figure
    vis_path = out / f"comparison_{split}_{scale}x_{Path(model_path).stem}.png"
    fig.tight_layout()
    fig.savefig(vis_path)
    plt.close(fig)
    logging.info(f"Saved comparison figure to {vis_path}")

    # Optionally print a table
    if show_table:
        print("\nFilename\tPSNR (dB)\tSSIM")
        for m in metrics_list:
            print(f"{m['filename']}\t{m['psnr']:.2f}\t{m['ssim']:.4f}")

    # Save metrics to JSON/CSV
    metrics_json = out / f"metrics_{split}_{scale}x_{Path(model_path).stem}.json"
    metrics_csv = out / f"metrics_{split}_{scale}x_{Path(model_path).stem}.csv"

    with open(metrics_json, 'w') as f_json:
        import json
        json.dump(metrics_list, f_json, indent=2)
    with open(metrics_csv, 'w', newline='') as f_csv:
        import csv
        writer = csv.DictWriter(f_csv, fieldnames=['filename', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(metrics_list)

    logging.info(f"Metrics saved to {metrics_json} and {metrics_csv}")

    # Optionally plot PSNR/SSIM curves
    if show_graphs and len(metrics_list) > 1:
        plot_paths = plot_metrics(metrics_list, Path(model_path).stem, split, scale, out)
        logging.info("Saved metric plots:")
        for name, p in plot_paths.items():
            logging.info(f"  - {name}: {p}")
        if verbose:
            for plot_name, plot_path in plot_paths.items():
                img = plt.imread(plot_path)
                plt.figure(figsize=(8, 5))
                plt.imshow(img)
                plt.title(plot_name.replace('_', ' ').title())
                plt.axis('off')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize SR Results")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory for train/val')
    parser.add_argument('--model_path', type=str, required=True, help='Path to checkpoint .pth file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'], help='Which split to use')
    parser.add_argument('--scale', type=int, default=4, choices=[2,3,4,8], help='Upscaling factor')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
    parser.add_argument('--show_diff', action='store_true', help='Show difference maps')
    parser.add_argument('--show_table', action='store_true', help='Print metrics table')
    parser.add_argument('--verbose', action='store_true', help='Display metric plots inline')
    parser.add_argument('--no-graphs', action='store_false', dest='show_graphs', help='Disable PSNR/SSIM plots')
    parser.add_argument('--light_blocks', action='store_true',
                        help='Use lightweight RRDB blocks (must match how the model was trained)')
    args = parser.parse_args()

    visualize(
        data_dir=args.data_dir,
        model_path=args.model_path,
        split=args.split,
        scale=args.scale,
        num_samples=args.num_samples,
        show_diff=args.show_diff,
        show_table=args.show_table,
        verbose=args.verbose,
        show_graphs=args.show_graphs,
        light_blocks=args.light_blocks
    )
