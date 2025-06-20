import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
import logging
from torch import nn
from torch.nn import functional as F

from rrdbnet import RRDBNet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _img_to_tensor(img: np.ndarray, bgr2rgb: bool = True, float32: bool = True) -> torch.Tensor:
    """Convert a BGR numpy image to a normalized PyTorch tensor.
    
    Args:
        img: Input image as numpy array (H×W×C or H×W)
        bgr2rgb: Convert from BGR to RGB if True
        float32: Convert to float32 normalized [0,1] if True, else uint8
        
    Returns:
        Torch tensor (1×C×H×W) in [0,1] or [0,255] range
    """
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    if img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=2)
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    tensor = torch.from_numpy(img)
    tensor = tensor.permute(2, 0, 1)  # HWC → CHW
    if float32:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.byte()
    return tensor.unsqueeze(0)  # add batch dim


def _tensor_to_img(tensor: torch.Tensor, rgb2bgr: bool = True, uint8: bool = True) -> np.ndarray:
    """Convert a tensor back to a numpy BGR image.
    
    Args:
        tensor: Input tensor (C×H×W or 1×C×H×W) in [-1,1] or [0,1]
        rgb2bgr: Convert output from RGB to BGR if True
        uint8: Convert to uint8 if True, else float32
        
    Returns:
        Numpy array (H×W×C) in BGR or RGB format
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    img = tensor.detach().cpu().clamp(-1.0, 1.0)
    img = (img + 1.0) / 2.0  # map from [-1,1] → [0,1]
    img = img.mul(255.0).round().byte().permute(1, 2, 0).numpy()  # CHW → HWC

    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def _load_model_weights(model: nn.Module, model_path: str, device: torch.device) -> Dict:
    """Helper function to load model weights from checkpoint.
    
    Args:
        model: Model to load weights into
        model_path: Path to model checkpoint
        device: Target device for loading
        
    Returns:
        Dictionary containing checkpoint information
        
    Raises:
        RuntimeError: If weights loading fails
    """
    logger.info(f"🔄 Loading weights from: {model_path}")
    try:
        # Disable weights_only to allow loading older checkpoint formats
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(ckpt, dict):
            # Handle different checkpoint formats
            if 'model_state' in ckpt:
                network_weights = ckpt['model_state']
            elif 'state_dict' in ckpt:
                network_weights = ckpt['state_dict']
            elif 'params' in ckpt:
                network_weights = ckpt['params']
            else:
                network_weights = ckpt
        else:
            network_weights = ckpt

        # Handle DataParallel/DP saved models
        network_weights = {k.replace('module.', ''): v for k, v in network_weights.items()}
        
        # Load state dict with strict=True to ensure all keys match
        model.load_state_dict(network_weights, strict=True)
        model.eval()
        logger.info(f"✅ Successfully loaded weights from {model_path}")
        
        return ckpt if isinstance(ckpt, dict) else {}
        
    except Exception as e:
        logger.error(f"❌ Failed to load model weights: {e}")
        raise RuntimeError(f"Model weights loading failed: {e}")


def load_sr_model(
    scale: int = 4,
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    use_light_blocks: bool = False,
    progressive_scale: bool = False,
    use_compile: bool = False,
    num_feat: Optional[int] = None,
    num_block: Optional[int] = None,
    num_grow_ch: Optional[int] = None
) -> nn.Module:
    """Load and initialize an RRDBNet super-resolution model.
    
    Args:
        scale: Upscaling factor (2, 3, 4, or 8)
        model_path: Path to model checkpoint (.pth file)
        device: Target device (default: auto-detect)
        use_light_blocks: Use lightweight RRDB blocks if True
        progressive_scale: Use progressive upsampling if True
        use_compile: Compile model with torch.compile() if True
        num_feat: Number of features (default: 64, or from checkpoint)
        num_block: Number of RRDB blocks (default: 23, or from checkpoint)
        num_grow_ch: Growth channels (default: 32, or from checkpoint)
        
    Returns:
        Initialized RRDBNet model in eval mode (if weights loaded)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"⚡ Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA: {torch.version.cuda}")
        logger.info(f"  PyTorch: {torch.__version__}")

    # Default model configuration
    default_num_feat = 64
    default_num_block = 23
    default_num_grow_ch = 32
    
    # If loading from checkpoint, try to extract model configuration
    if model_path is not None:
        try:
            # Peek into checkpoint to get configuration
            ckpt = torch.load(str(model_path), map_location='cpu', weights_only=False)
            
            if isinstance(ckpt, dict) and 'model_config' in ckpt:
                # Use saved configuration if available
                config = ckpt['model_config']
                num_feat = num_feat or config.get('num_feat', default_num_feat)
                num_block = num_block or config.get('num_block', default_num_block)
                num_grow_ch = num_grow_ch or config.get('num_grow_ch', default_num_grow_ch)
                # Override these if they were saved
                if 'use_light_blocks' in config:
                    use_light_blocks = config['use_light_blocks']
                if 'progressive_scale' in config:
                    progressive_scale = config['progressive_scale']
                logger.info(f"📋 Loaded model configuration from checkpoint")
            else:
                # Try to infer from state dict keys
                if isinstance(ckpt, dict):
                    state_dict = ckpt.get('model_state', ckpt.get('state_dict', ckpt.get('params', ckpt)))
                else:
                    state_dict = ckpt
                    
                # Count RRDB blocks from state dict keys
                rrdb_keys = [k for k in state_dict.keys() if 'body.' in k and '.rdb1.' in k]
                if rrdb_keys:
                    inferred_blocks = max(int(k.split('.')[1]) for k in rrdb_keys) + 1
                    num_block = num_block or inferred_blocks
                    logger.info(f"📊 Inferred {inferred_blocks} RRDB blocks from checkpoint")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load configuration from checkpoint: {e}")
    
    # Use defaults if not set
    num_feat = num_feat or default_num_feat
    num_block = num_block or default_num_block
    num_grow_ch = num_grow_ch or default_num_grow_ch

    logger.info(f"🔧 Model configuration: feat={num_feat}, blocks={num_block}, grow={num_grow_ch}, light={use_light_blocks}")

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=scale,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
        use_light_blocks=use_light_blocks,
        progressive_scale=progressive_scale
    ).to(device)

    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
        logger.info("🔧 Model compiled with torch.compile()")

    if model_path is not None:
        _load_model_weights(model, str(model_path), device)

    return model


def save_sr_model(
    model: nn.Module,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    best_metrics: Optional[Dict] = None,
    model_config: Optional[Dict] = None
):
    """Save model checkpoint with configuration.
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        best_metrics: Optional best metrics dictionary
        model_config: Optional model configuration to save
    """
    # Extract model configuration if not provided
    if model_config is None:
        # Try to extract from model
        if hasattr(model, 'module'):  # DataParallel
            base_model = model.module
        else:
            base_model = model
            
        model_config = {
            'num_feat': base_model.conv_first.out_channels,
            'num_block': len(base_model.body),
            'num_grow_ch': 32,  # Default, hard to infer
            'scale': base_model.scale,
            'use_light_blocks': hasattr(base_model.body[0].rdb1, 'attention'),
            'progressive_scale': base_model.progressive_scale
        }
    
    checkpoint = {
        'model_state': model.state_dict(),
        'model_config': model_config,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state'] = optimizer.state_dict()
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if best_metrics is not None:
        checkpoint['best_metrics'] = best_metrics
        
    torch.save(checkpoint, save_path)
    logger.info(f"💾 Saved checkpoint to {save_path}")


def predict_sr(model: nn.Module, img_bgr: np.ndarray) -> np.ndarray:
    """Run super-resolution prediction on an input BGR image.
    
    Args:
        model: Loaded RRDBNet model
        img_bgr: Input image (H×W×3) in BGR format (uint8)
        
    Returns:
        Super-resolved image (H×W×3) in BGR format (uint8)
    """
    device = next(model.parameters()).device
    model.eval()

    # Convert to tensor and normalize to [-1, 1]
    img_tensor = _img_to_tensor(img_bgr).to(device)
    img_tensor = img_tensor * 2.0 - 1.0

    with torch.no_grad():
        sr_tensor = model(img_tensor)
        
    return _tensor_to_img(sr_tensor, rgb2bgr=True, uint8=True)