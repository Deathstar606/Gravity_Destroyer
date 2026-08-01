#!/usr/bin/env python3
"""
XPU Hardware Test Script for DINGO-T1
=====================================

This script:
1. Initializes Intel Arc GPU (XPU) with proper environment settings
2. Loads pre-trained DINGO-T1 weights from Zenodo
3. Freezes all but the last 2 transformer attention blocks
4. Runs a short test training session (11K waveforms)
5. Validates model on XPU hardware

Usage:
------
python train_xpu_hardware_test.py \\
    --config train_settings_xpu_hardware_test.yaml \\
    --zenodo_checkpoint /path/to/dingo_t1.pt \\
    --output_dir ./xpu_test_run

Author: Transfer Learning Test Suite
Date: 2026-06-19
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch.optim as optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up XPU environment BEFORE importing torch
def setup_xpu_environment():
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
import yaml
import numpy as np
from typing import Dict, Tuple

# Import dingo modules
from dingo.core.posterior_models.build_model import build_model_from_kwargs, autocomplete_model_kwargs
from dingo.core.utils import build_train_and_test_loaders
from dingo.gw.dataset import WaveformDataset
from dingo.gw.training.train_builders import build_dataset, set_train_transforms

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


def load_config(config_path: str) -> Dict:
    """
    Load YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to YAML config file
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"✓ Configuration loaded")
    return config


def load_pretrained_weights(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """
    Load pre-trained DINGO-T1 weights from Zenodo checkpoint.
    
    Parameters
    ----------
    model : nn.Module
        Model to load weights into
    checkpoint_path : str
        Path to pre-trained checkpoint
    device : torch.device
        Device to load checkpoint to
    """
    logger.info(f"Loading pre-trained weights from {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both direct state_dict and wrapped state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'posterior_model' in checkpoint:
            state_dict = checkpoint['posterior_model']
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to allow for architectural changes
        missing, unexpected = model.network.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys in checkpoint: {len(missing)} keys")
            for key in missing[:5]:  # Show first 5
                logger.debug(f"  - {key}")
        
        if unexpected:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected)} keys")
            for key in unexpected[:5]:  # Show first 5
                logger.debug(f"  - {key}")
        
        logger.info("✓ Pre-trained weights loaded successfully")
        
    except FileNotFoundError:
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading checkpoint 😔: {e}")
        raise


def freeze_all_except_last_n_layers(model: nn.Module, n: int = 2) -> None:
    """
    Freeze all parameters except the last n transformer layers.
    This enables efficient transfer learning on limited hardware.
    
    Parameters
    ----------
    model : nn.Module
        Model to freeze
    n : int
        Number of layers from the end to keep unfrozen (default: 2)
    """
    logger.info(f"Freezing all layers except the last {n} transformer blocks...")
    
    # Use the method we added to TransformerModel
    embedding_net = model.network.embedding_net
    
    if hasattr(embedding_net, 'freeze_all_except_last_n_layers'):
        embedding_net.freeze_all_except_last_n_layers(n=n)
        logger.info(f"✓ Froze all except last {n} layers using TransformerModel.freeze_all_except_last_n_layers()")
    else:
        # Fallback: manual freezing
        logger.warning("freeze_all_except_last_n_layers method not found, using manual freezing")
        for param in model.network.parameters():
            param.requires_grad = False
        
        # Unfreeze NSF parameters (we want these to train)
        if hasattr(model, 'flow'):
            for param in model.network.flow.parameters():
                param.requires_grad = True
    
    # Print frozen status
    if hasattr(embedding_net, 'get_frozen_status'):
        status = embedding_net.get_frozen_status()
        logger.info("\nFrozen status summary:")
        total_frozen = 0
        total_trainable = 0
        for component, counts in status.items():
            trainable = counts['trainable']
            frozen = counts['frozen']
            total_frozen += frozen
            total_trainable += trainable
            status_str = f"  {component:30s}: trainable={trainable:>10,d}, frozen={frozen:>10,d}"
            logger.info(status_str)
        logger.info(f"  {'TOTAL':30s}: trainable={total_trainable:>10,d}, frozen={total_frozen:>10,d}")
        logger.info(f"  Freezing efficiency: {100*total_frozen/(total_frozen+total_trainable):.1f}% frozen")


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """
    Count trainable and frozen parameters in model.
    
    Parameters
    ----------
    model : nn.Module
        Model to count parameters for
        
    Returns
    -------
    Tuple[int, int, int]
        (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.network.parameters())
    trainable_params = sum(p.numel() for p in model.network.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return total_params, trainable_params, frozen_params


def log_model_summary(model: nn.Module, config: Dict) -> None:
    """
    Log comprehensive model and hardware summary.
    
    Parameters
    ----------
    model : nn.Module
        Model to summarize
    config : Dict
        Configuration dictionary
    """
    total, trainable, frozen = count_parameters(model)
    
    logger.info("\n" + "="*70)
    logger.info("DINGO-T1 HARDWARE TEST - MODEL SUMMARY")
    logger.info("="*70)
    logger.info(f"Total parameters:     {total:,d}")
    logger.info(f"Trainable parameters: {trainable:,d} ({100*trainable/total:.2f}%)")
    logger.info(f"Frozen parameters:    {frozen:,d} ({100*frozen/total:.2f}%)")
    logger.info(f"\nBatch size:           {config['training']['stage_0']['batch_size']}")
    logger.info(f"Num epochs:           {config['training']['stage_0']['epochs']}")
    logger.info(f"NSF flow steps:       {config['model']['posterior_kwargs']['num_flow_steps']}")
    logger.info(f"Device:               {torch.device('xpu' if torch.xpu.is_available() else 'cpu')}")
    logger.info("="*70 + "\n")


def main(args):
    """
    Main training function.
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging to file
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger.info("="*70)
    logger.info("DINGO-T1 XPU HARDWARE TEST")
    logger.info(f"Start time: {datetime.now().isoformat()}")
    logger.info("="*70)
    
    # Get device
    device, device_type = get_device()
    
    # Load configuration
    config = load_config(args.config)
    
    # Build model from config
    logger.info("Building model from configuration...")
    
    # If a checkpoint is provided, build the model from the file; otherwise build from settings
    if args.zenodo_checkpoint:
        model = build_model_from_kwargs(filename=args.zenodo_checkpoint, device=device, print_output=True)
        # model.to(device)  <-- DELETE THIS LINE
        load_pretrained_weights(model, args.zenodo_checkpoint, device)
    else:
        # For testing without data: inject dummy tokenizer and embedding dimensions
        if "embedding_kwargs" in config["model"]:
            emb_kwargs = config["model"]["embedding_kwargs"]
            d_model = emb_kwargs.get("transformer_kwargs", {}).get("d_model", 1024)

            # 1. Inject Tokenizer
            if "tokenizer_kwargs" in emb_kwargs:
                if "input_dims" not in emb_kwargs["tokenizer_kwargs"]:
                    emb_kwargs["tokenizer_kwargs"]["input_dims"] = [128, 256]
                    logger.info("  Injected dummy tokenizer input_dims: [128, 256]")
                if "output_dim" not in emb_kwargs["tokenizer_kwargs"]:
                    emb_kwargs["tokenizer_kwargs"]["output_dim"] = d_model
                    logger.info(f"  Injected dummy tokenizer output_dim: {d_model}")
            
            # 2. Inject Final Net
            if "final_net_kwargs" in emb_kwargs:
                if "input_dim" not in emb_kwargs["final_net_kwargs"]:
                    emb_kwargs["final_net_kwargs"]["input_dim"] = d_model
                    logger.info(f"  Injected dummy final_net input_dim: {d_model}")
                    
        # 3. Inject Posterior Constraints (THE NEW FIX)
        if "posterior_kwargs" in config["model"]:
            post_kwargs = config["model"]["posterior_kwargs"]
            if "input_dim" not in post_kwargs:
                post_kwargs["input_dim"] = 15  # Standard GW parameter count
                logger.info("  Injected dummy posterior input_dim: 15")
            if "context_dim" not in post_kwargs:
                # Match the final_net output_dim from your YAML (128)
                final_out = config["model"].get("embedding_kwargs", {}).get("final_net_kwargs", {}).get("output_dim", 128)
                post_kwargs["context_dim"] = final_out
                logger.info(f"  Injected dummy posterior context_dim: {final_out}")
        
        # Build the model and route the device correctly
        settings = {"train_settings": config}
        
        model = build_model_from_kwargs(settings=settings, device=device)
        model.network.to(device)
        logger.info("Raw model built successfully without Zenodo weights.")
    
    if not args.zenodo_checkpoint:
        logger.warning("No zenodo checkpoint provided, training from random initialization")
    
    # Freeze all but last 2 layers
    freeze_all_except_last_n_layers(model, n=2)
    
    # Log model summary
    log_model_summary(model, config)
    
    # Ensure an optimizer is defined before the loop
    optimizer = optim.Adam(model.network.parameters(), lr=1e-4)

    logger.info("Testing full 5-epoch loop with simulated batches...")
    try:
        dataset_size = 192
        batch_size = 64
        num_epochs = 5
        num_batches = dataset_size // batch_size  # 192 / 64 = 3 batches per epoch
        
        num_tokens = 128
        num_features = 48 
        num_params = 14
        
        logger.info(f"Dataset Size: {dataset_size} | Batch Size: {batch_size} | Total Epochs: {num_epochs}")
        
        for epoch in range(num_epochs):
            logger.info(f"\n========== STARTING EPOCH {epoch + 1}/{num_epochs} ==========")
            model.network.train() # Lock network into training mode
            
            running_loss = 0.0
            
            for batch_idx in range(num_batches):
                # 1. Generate the batch data on-the-fly (Mimics DataLoader streaming)
                features = torch.randn(batch_size, num_tokens, num_features, device=device)
                positions = torch.ones((batch_size, num_tokens, 3), dtype=torch.long, device=device)
                padding_mask = torch.zeros((batch_size, num_tokens), dtype=torch.bool, device=device)
                y = torch.randn(batch_size, num_params, device=device)
                
                # 2. Reset gradients
                optimizer.zero_grad()
                
                # 3. Forward Pass
                log_prob, logging_info = model.network(y, features, positions, padding_mask)
                
                # IMPORTANT: Normalizing Flows maximize log probability. 
                # Therefore, we must minimize NEGATIVE log probability.
                loss = -log_prob.mean() 
                
                # 4. Backward Pass
                loss.backward()
                
                # 5. Weight Update
                optimizer.step()
                
                running_loss += loss.item()
                logger.info(f"  Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
                
            # Epoch Summary
            avg_epoch_loss = running_loss / num_batches
            logger.info(f"✓ Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        # Post-Training Gradient Verification
        grad_count = sum(1 for p in model.network.parameters() if p.requires_grad and p.grad is not None)
        logger.info(f"\n✓ 5-Epoch loop successful")
        logger.info(f"  Parameters actively trained on final batch: {grad_count}")
        
    except Exception as e:
        logger.error(f"✗ Error during model test: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("\n" + "="*70)
    logger.info("HARDWARE TEST COMPLETE")
    logger.info(f"Log file: {log_file}")
    logger.info("="*70)
    
    logger.info("\n" + "="*70)
    logger.info("HARDWARE TEST COMPLETE")
    logger.info(f"Log file: {log_file}")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("1. If forward/backward passes succeeded, you can proceed with full training")
    logger.info("2. Use dingo_pipe with the XPU config to run actual training:")
    logger.info(f"   dingo_pipe {args.config}")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DINGO-T1 XPU Hardware Test - Validate setup before full training"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--zenodo_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained DINGO-T1 checkpoint from Zenodo (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./xpu_test_run",
        help="Output directory for test results and logs"
    )
    
    args = parser.parse_args()
    main(args)