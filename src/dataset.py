import os
import glob
import random
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def fast_loader(path: str) -> Image.Image:
    """Faster image loading using OpenCV with BGR to RGB conversion."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


class SRDataset(Dataset):
    """
    Optimized Super-Resolution Dataset for RRDBNet training with standardized sizes.
    Features faster image loading, LR caching, file list caching, and consistent output dimensions.

    Directory structure:
        root_dir/
          train/
            hr/       # high-res originals
            lr/       # optional: precomputed low-res
          val/
            hr/
            lr/
    """

    _filelist_cache = {}  # Class-level cache for file paths

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        scale: int = 4,
        hr_size: tuple = (256, 256),  # Standard HR output size
        patch_size: int = None,        # None for full standardized size
        cache: bool = True,           # Cache LR images in RAM
    ):
        """
        Args:
            root_dir: path to dataset root
            split: 'train' or 'val'
            scale: upsampling factor
            hr_size: target size for HR images (height, width)
            patch_size: HR patch size for random crops (None = use full hr_size)
            cache: cache LR images in memory
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.scale = scale
        self.hr_size = hr_size
        self.lr_size = (hr_size[0] // scale, hr_size[1] // scale)
        self.patch_size = patch_size
        self.cache = cache
        self._cached_lr = {}  # LR image cache

        # Validate patch size
        if patch_size is not None:
            if patch_size > hr_size[0] or patch_size > hr_size[1]:
                raise ValueError(f"Patch size {patch_size} cannot be larger than HR size {hr_size}")

        # Get file paths (use cached if available)
        cache_key = f"{root_dir}_{split}"
        if cache_key in SRDataset._filelist_cache:
            self.hr_paths, self.lr_paths, self.use_lr = SRDataset._filelist_cache[cache_key]
        else:
            # High-res directory
            self.hr_dir = self.root_dir / split / 'hr'
            if not self.hr_dir.exists():
                raise FileNotFoundError(f"HR folder not found: {self.hr_dir}")
            
            self.hr_paths = sorted(glob.glob(str(self.hr_dir / '*')))
            if not self.hr_paths:
                raise ValueError(f"No images found in: {self.hr_dir}")

            # Low-res directory (optional)
            self.lr_dir = self.root_dir / split / 'lr'
            if self.lr_dir.exists() and any(self.lr_dir.iterdir()):
                self.use_lr = True
                self.lr_paths = sorted(glob.glob(str(self.lr_dir / '*')))
                if len(self.lr_paths) != len(self.hr_paths):
                    raise ValueError("LR and HR counts do not match")
            else:
                self.use_lr = False
                self.lr_paths = []
            
            # Cache the file lists
            SRDataset._filelist_cache[cache_key] = (self.hr_paths, self.lr_paths, self.use_lr)

    def __len__(self):
        return len(self.hr_paths)

    def _load_and_preprocess(self, idx: int):
        """Load and preprocess HR and LR images with caching support."""
        # Check cache first
        if idx in self._cached_lr:
            return self._cached_lr[idx]

        # Load HR image
        hr = fast_loader(self.hr_paths[idx])
        
        # Resize HR to standard size before generating LR
        hr = hr.resize(self.hr_size, Image.BICUBIC)

        # Generate or load LR
        if self.use_lr:
            lr = fast_loader(self.lr_paths[idx])
            lr = lr.resize(self.lr_size, Image.BICUBIC)
        else:
            lr = hr.resize(self.lr_size, Image.BICUBIC)

        # Cache if enabled
        if self.cache:
            self._cached_lr[idx] = (hr, lr)

        return hr, lr

    def __getitem__(self, idx: int):
        hr, lr = self._load_and_preprocess(idx)

        # Apply random patch crop during training if specified
        if self.patch_size and self.split == 'train':
            # HR crop
            ph = self.patch_size
            max_y = hr.height - ph
            max_x = hr.width - ph
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            hr = TF.crop(hr, y, x, ph, ph)
            
            # Corresponding LR crop
            lp_size = ph // self.scale
            lr = TF.crop(lr, y // self.scale, x // self.scale, lp_size, lp_size)

        # Convert to tensor [0,1]
        lr_tensor = TF.to_tensor(lr)
        hr_tensor = TF.to_tensor(hr)
        return lr_tensor, hr_tensor