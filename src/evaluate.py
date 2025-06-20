#!/usr/bin/env python3
import torch
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import warnings

# Try both import styles so users can call from project root or from src/
try:
    from dataset import SRDataset
    from model import load_sr_model
except ImportError:
    from src.dataset import SRDataset
    from src.model import load_sr_model

warnings.filterwarnings("ignore", category=UserWarning)


def calculate_metrics(hr_img, sr_img):
    """
    Compute PSNR & SSIM between two UINT8 numpy arrays (or PIL Images).
    If SSIM's default window is too large for the image, pick the largest
    odd win_size ≤ min(height, width). If that still fails, set SSIM=0.
    """
    hr_arr = np.array(hr_img) if not isinstance(hr_img, np.ndarray) else hr_img
    sr_arr = np.array(sr_img) if not isinstance(sr_img, np.ndarray) else sr_img

    hr_arr = np.clip(hr_arr, 0, 255).astype(np.uint8)
    sr_arr = np.clip(sr_arr, 0, 255).astype(np.uint8)

    # PSNR expects floats, data_range=255
    psnr_val = compute_psnr(hr_arr, sr_arr, data_range=255)

    # Determine if color or grayscale
    if hr_arr.ndim == 3 and hr_arr.shape[2] == 3:
        multichannel = True
    else:
        multichannel = False

    # Attempt SSIM with default window
    try:
        ssim_val = compute_ssim(
            hr_arr,
            sr_arr,
            multichannel=multichannel,
            data_range=255
        )
    except ValueError as e:
        # Likely "win_size exceeds image extent"
        # Pick a smaller odd window: largest odd ≤ min(height, width)
        h, w = hr_arr.shape[:2]
        min_dim = min(h, w)
        # largest odd integer ≤ min_dim
        win_size = min_dim if (min_dim % 2 == 1) else (min_dim - 1)
        if win_size < 1:
            # Too small for SSIM
            return psnr_val, 0.0

        try:
            ssim_val = compute_ssim(
                hr_arr,
                sr_arr,
                multichannel=multichannel,
                data_range=255,
                win_size=win_size
            )
        except Exception:
            ssim_val = 0.0

    return psnr_val, ssim_val


def evaluate(
    data_dir: str,
    model_path: str,
    scale: int = 4,
    batch_size: int = 4,
    num_workers: int = 4,
    save_results: bool = False,
    output_dir: str = "results",
    use_light_blocks: bool = False,
    progressive_scale: bool = False
) -> dict:
    """
    Evaluate a super-resolution model on the validation set.

    - data_dir: contains 'val/hr' and 'val/lr'
    - model_path: path to the checkpoint .pth file
    - scale: upscaling factor
    - batch_size, num_workers: for DataLoader
    - save_results: if True, saves SR images under output_dir
    - use_light_blocks, progressive_scale: must match training flags
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n⚙️ Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  PyTorch: {torch.__version__}")

    print("\n🔄 Loading model.")
    model = load_sr_model(
        scale=scale,
        model_path=model_path,
        device=device,
        use_light_blocks=use_light_blocks,
        progressive_scale=progressive_scale
    )
    model.eval()

    val_ds = SRDataset(root_dir=data_dir, split="val", scale=scale, patch_size=None)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n📊 Validation set: {len(val_ds)} images.")
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Will save SR results to: {output_dir}")

    psnr_vals = []
    ssim_vals = []
    inference_times = []

    print(f"\n🔍 Evaluating on {len(val_ds)} images (batch size: {batch_size})...")
    try:
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(tqdm(val_loader, desc="Processing")):
            lr_imgs = lr_imgs.to(device, non_blocking=True)
            hr_imgs = hr_imgs.to(device, non_blocking=True)

            # Time the forward pass
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            with torch.no_grad():
                sr_raw = model(lr_imgs)                    # → in [-1 .. +1]
                sr_imgs = (sr_raw + 1.0) / 2.0             # map to [0..1]
                sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)

            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - start_time)

            # Convert SR images to uint8 numpy arrays
            sr_imgs_cpu = (sr_imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)
            
            # Convert HR images from [-1..+1] to [0..1] then to uint8 numpy arrays
            hr_01 = (hr_imgs + 1.0) / 2.0
            hr_imgs_cpu = (hr_01.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).round().astype(np.uint8)

            for i in range(sr_imgs_cpu.shape[0]):
                sr_uint8 = sr_imgs_cpu[i]
                hr_uint8 = hr_imgs_cpu[i]

                psnr_val, ssim_val = calculate_metrics(hr_uint8, sr_uint8)
                psnr_vals.append(psnr_val)
                ssim_vals.append(ssim_val)

                if save_results:
                    img_name = Path(val_ds.hr_paths[batch_idx * batch_size + i]).stem
                    out_path = Path(output_dir) / f"{img_name}_SR.png"
                    Image.fromarray(sr_uint8).save(str(out_path))

    except RuntimeError as e:
        print(f"\n❌ Evaluation failed: {e}")
        if "CUDA out of memory" in str(e):
            print("  Try reducing batch size or patch size.")
        raise

    avg_psnr = float(np.mean(psnr_vals)) if psnr_vals else 0.0
    avg_ssim = float(np.mean(ssim_vals)) if ssim_vals else 0.0
    avg_time_ms = float(np.mean(inference_times)) * 1000 if inference_times else 0.0

    print("\n📈 Evaluation Results:")
    print(f"  PSNR: {avg_psnr:.2f} dB ± {np.std(psnr_vals):.2f}")
    print(f"  SSIM: {avg_ssim:.4f} ± {np.std(ssim_vals):.4f}")
    print(f"  Inference speed: {avg_time_ms:.2f} ms/image ± {np.std(inference_times)*1000:.2f} ms")
    print(f"  Total time: {sum(inference_times):.2f}s for {len(val_ds)} images")

    return {
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "inference_time_ms": avg_time_ms
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Super-Resolution Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing train/val subfolders"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to checkpoint (.pth) file"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[2, 3, 4, 8],
        help="Upscaling factor"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save output SR images to disk"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory in which to save SR images"
    )
    parser.add_argument(
        "--light_blocks",
        action="store_true",
        help="Use lightweight RRDB blocks (must match training)"
    )
    parser.add_argument(
        "--progressive",
        action="store_true",
        help="Use progressive upscaling (must match training)"
    )

    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        model_path=args.model_path,
        scale=args.scale,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_results=args.save_results,
        output_dir=args.output_dir,
        use_light_blocks=args.light_blocks,
        progressive_scale=args.progressive
    )