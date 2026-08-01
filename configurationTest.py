import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up XPU environment BEFORE importing torch
def setup_xpu_environment():
    print("Setting up XPU environment variables for optimal performance...")
    """
    Initialize Intel Arc GPU environment variables for optimal performance.
    These must be set BEFORE any torch imports.
    """
    # Enable expandable memory segments for flexible allocation
    os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
    
    # Enable persistent SYCL cache for faster compilation
    os.environ["SYCL_CACHE_PERSISTENT"] = "1"
    
    # Enable relaxed allocation limits for better memory handling
    os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
    
    # Disable TF32 for reproducibility (matching zenodo training)
    os.environ["TORCH_ALLOW_TF32"] = "0"
    
    logging.info("XPU environment variables configured")


# Call setup BEFORE torch import
setup_xpu_environment()

import torch
import torch.nn as nn
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device() -> Tuple[torch.device, str]:
    """
    Detect and initialize appropriate device (XPU preferred, fallback to CPU).
    
    Returns
    -------
    Tuple[torch.device, str]
        Device object and device type string
    """
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu:0")
        device_type = "xpu"
        logger.info("✓ XPU device detected and initialized")
        logger.info(f"  GPU: {torch.xpu.get_device_name()}")
        
        # Log XPU properties
        try:
            props = torch.xpu.get_device_properties()
            logger.info(f"  Max work group size: {props.get('max_work_group_size', 'N/A')}")
            logger.info(f"  Max compute units: {props.get('max_compute_units', 'N/A')}")
        except Exception as e:
            logger.debug(f"Could not read full device properties: {e}")
            
    else:
        logger.warning("⚠ XPU not available, falling back to CPU")
        device = torch.device("cpu")
        device_type = "cpu"
        logger.info("  Using CPU device")
    
    return device, device_type

if __name__ == "__main__":
    logger.info("Starting XPU environment test...")
    # Detect device and fetch properties
    device, device_type = get_device()
    #print(f"Using device: {device} ({device_type.upper()})")
    # Perform a simple tensor allocation to verify the device is fully functional
    try:
        logger.info(f"Testing tensor allocation and computation on {device_type.upper()}...")
        test_tensor = torch.ones((1000, 1000), device=device)
        result_tensor = test_tensor @ test_tensor  # Simple matrix multiplication
        
        logger.info("✓ Tensor operation successful!")
        logger.info(f"  Test tensor shape: {result_tensor.shape}")
        
        # Clear VRAM 
        if device_type == "xpu":
            torch.xpu.empty_cache()
            
    except Exception as e:
        logger.error(f"✗ Tensor operation failed: {e}")
        
    logger.info("Test execution completed.")