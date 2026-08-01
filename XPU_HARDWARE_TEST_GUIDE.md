# DINGO-T1 XPU Hardware Test Guide

## Overview

This hardware test setup validates your Intel Arc A750 GPU configuration before running full training on DINGO-T1. The test includes:

✅ XPU device initialization  
✅ Pre-trained weight loading from Zenodo  
✅ Automatic freezing of layers 0-5 (keep last 2 unfrozen)  
✅ Forward/backward pass validation  
✅ Memory and compute verification

---

## Files Created

### 1. **transformer.py (modified)**

Added three methods to `TransformerModel` class:

- `freeze_all_except_last_n_layers(n=2)` — Freeze all but last n transformer blocks
- `unfreeze_all_layers()` — Unfreeze entire model
- `get_frozen_status()` — Get detailed frozen/trainable parameter counts

**Location**: `dingo/dingo/core/nn/transformer.py`  
**Lines added**: ~130 lines

### 2. **train_settings_xpu_hardware_test.yaml**

Optimized configuration for your hardware:

- **NSF**: 15 flow steps (down from 30), 256D hidden (down from 512)
- **Batch size**: 64 (down from 8192 for single GPU)
- **Training epochs**: 50 (quick test)
- **Device**: XPU
- **Other settings**: Matches zenodo architecture (tokenizer, transformer, etc.)

**Location**: `dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml`

### 3. **train_xpu_hardware_test.py**

Standalone test script that:

- Sets XPU environment variables (BEFORE torch import) ✓
- Detects Intel Arc GPU
- Loads Zenodo checkpoint
- Freezes layers 0-5
- Tests forward/backward passes
- Validates gradient flow

**Location**: `./train_xpu_hardware_test.py` (project root)

---

## Pre-Flight Checklist

Before running the test, verify:

- [ ] You've installed XPU stack:

  ```bash
  pip install torch==2.11.0+xpu torchvision==0.26.0+xpu torchaudio==2.11.0+xpu --index-url https://download.pytorch.org/whl/xpu
  pip install intel-cmplr-lib-rt==2025.3.2 intel-opencl-rt==2025.3.2 intel-sycl-rt==2025.3.2
  ```

- [ ] DINGO installed from source:

  ```bash
  cd dingo
  pip install -e ".[dev]"
  ```

- [ ] You have the 11,000 waveform dataset:

  ```bash
  ls -lh dingo-T1/01_paper_settings/01_training/01_waveform_dataset/waveform_dataset.hdf5
  ```

- [ ] You have the ASD dataset:

  ```bash
  ls -lh dingo-T1/01_paper_settings/01_training/02_asd_dataset/asds_O3.hdf5
  ```

- [ ] Zenodo checkpoint downloaded (optional but recommended):
  ```bash
  cd dingo-T1/02_inference_with_pretrained_models
  wget -O dingo_t1.pt "https://zenodo.org/records/17726076/files/dingo_t1.pt?download=1"
  ```

---

## Running the Hardware Test

### Quick Test (No Pre-trained Weights)

```bash
cd /path/to/dingorep

python train_xpu_hardware_test.py \
    --config dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml \
    --output_dir ./xpu_test_results
```

### Full Test (With Pre-trained Weights)

```bash
cd /path/to/dingorep

python train_xpu_hardware_test.py \
    --config dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml \
    --zenodo_checkpoint dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt \
    --output_dir ./xpu_test_results
```

---

## Expected Output

### Successful Test Output

```
======================================================================
DINGO-T1 HARDWARE TEST
Start time: 2026-06-19T15:30:45.123456
======================================================================

2026-06-19 15:30:45 - __main__ - INFO - XPU environment variables configured
2026-06-19 15:30:47 - __main__ - INFO - ✓ XPU device detected and initialized
  GPU: Intel(R) Arc(TM) A750 Graphics
  Max work group size: ...

2026-06-19 15:30:48 - __main__ - INFO - ✓ Configuration loaded
2026-06-19 15:30:50 - __main__ - INFO - Building model from configuration...
2026-06-19 15:30:52 - __main__ - INFO - ✓ Pre-trained weights loaded successfully
2026-06-19 15:30:53 - __main__ - INFO - Freezing all layers except the last 2 transformer blocks...
2026-06-19 15:30:53 - __main__ - INFO - ✓ Froze all except last 2 layers using TransformerModel.freeze_all_except_last_n_layers()

Frozen status summary:
  tokenizer:              trainable=         0, frozen=    123,456
  transformer_layer_0:    trainable=         0, frozen= 1,234,567
  transformer_layer_1:    trainable=         0, frozen= 1,234,567
  ...
  transformer_layer_6:    trainable= 2,345,678, frozen=         0
  transformer_layer_7:    trainable= 2,345,678, frozen=         0
  final_net:              trainable=         0, frozen=     98,765
  Freezing efficiency: 87.5% frozen

======================================================================
DINGO-T1 HARDWARE TEST - MODEL SUMMARY
======================================================================
Total parameters:     50,123,456
Trainable parameters: 4,691,356 (9.4%)
Frozen parameters:    45,432,100 (90.6%)

Batch size:           64
Num epochs:           50
NSF flow steps:       15
Device:               xpu:0
======================================================================

2026-06-19 15:30:55 - __main__ - INFO - Testing forward pass with dummy data...
2026-06-19 15:30:58 - __main__ - INFO - ✓ Forward pass successful
  Output shape: torch.Size([4, 14])
  Logging info: {'num_tokens': 128, 'num_all_tokens': 128}

2026-06-19 15:31:00 - __main__ - INFO - Testing backward pass (gradient flow)...
2026-06-19 15:31:03 - __main__ - INFO - ✓ Backward pass successful
  Parameters with gradients: 2  # Only layers 6-7 have gradients

======================================================================
HARDWARE TEST COMPLETE
Log file: ./xpu_test_results/training_20260619_153103.log
======================================================================

Next steps:
1. If forward/backward passes succeeded, you can proceed with full training
2. Use dingo_pipe with the XPU config to run actual training:
   dingo_pipe dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml
```

---

## What to Check in Output

### ✅ SUCCESS Indicators

1. **XPU detected**: Should see `✓ XPU device detected and initialized`
2. **Freezing efficiency**: Should show ~87-90% of parameters frozen
3. **Trainable parameters**: Should be ~4-5M (only last 2 transformer blocks)
4. **Forward pass**: `✓ Forward pass successful`
5. **Backward pass**: `✓ Backward pass successful` with gradients flowing
6. **Gradient count**: Should be small (only unfrozen layers have gradients)

### 🔴 FAILURE Indicators (and fixes)

| Error                  | Cause                 | Fix                                                    |
| ---------------------- | --------------------- | ------------------------------------------------------ |
| `XPU not available`    | XPU not installed     | Install XPU stack: `pip install torch==2.11.0+xpu ...` |
| `CUDA out of memory`   | GPU memory too small  | Reduce batch_size in YAML (already at 64)              |
| `Missing checkpoint`   | Zenodo file not found | Download: `wget https://zenodo.org/...`                |
| `Backward pass failed` | Freezing error        | Check transformer.py modifications                     |
| `0% trainable params`  | All frozen (wrong)    | Check freeze_all_except_last_n_layers()                |

---

## Next Steps After Successful Test

### If Forward/Backward Pass Succeeds ✅

**Option 1: Run Full Training (Recommended)**

```bash
# Copy config to standard location
cp dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml \
   dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml

# Run training via dingo_pipe
cd dingo-T1/01_paper_settings/01_training/03_training
dingo_pipe train_settings.yaml
```

**Option 2: Monitor Training**

- Check logs in `./xpu_test_results/`
- Watch VRAM usage: `watch nvidia-smi` (or Intel GPU equivalent)
- Expected VRAM: 8-12GB for batch_size=64

### Memory Optimization Tips

If you run into memory issues during full training:

1. **Reduce batch size** (in YAML):

   ```yaml
   training:
     stage_0:
       batch_size: 32 # From 64
   ```

2. **Reduce NSF complexity** (in YAML):

   ```yaml
   model:
     posterior_kwargs:
       num_flow_steps: 10 # From 15
       base_transform_kwargs:
         hidden_dim: 128 # From 256
         num_transform_blocks: 2 # From 3
   ```

3. **Disable gradient accumulation** (in YAML):
   ```yaml
   training:
     stage_0:
       gradient_updates_per_optimizer_step: 1 # From 2
   ```

---

## XPU-Specific Notes

### Environment Variables (Already Set in Script)

```python
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SYCL_CACHE_PERSISTENT"] = "1"
os.environ["UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS"] = "1"
os.environ["TORCH_ALLOW_TF32"] = "0"
```

These are set **before** torch import in the test script. If you use your own training loop, add these at the very top.

### Known XPU Quirks

- First run compiles kernels (slow, 1-2 min)
- Subsequent runs are faster
- Memory allocation is "lazy" (don't worry about initial warnings)
- Watch for `SYCL` warnings in logs (usually harmless)

---

## Configuration Details: YAML vs. Zenodo

### What's Different (XPU Test vs. Zenodo)

| Setting       | Zenodo        | XPU Test     | Reason            |
| ------------- | ------------- | ------------ | ----------------- |
| NSF steps     | 30            | 15           | Memory savings    |
| NSF hidden    | 512           | 256          | Memory savings    |
| NSF blocks    | 5             | 3            | Memory savings    |
| Batch size    | 8192          | 64           | Single GPU        |
| Num epochs    | 300           | 50           | Quick test        |
| Frozen layers | 0 (all train) | 0-5 (frozen) | Transfer learning |
| Device        | cuda          | xpu          | Hardware support  |

### What's the Same (Preserved from Zenodo)

- Tokenizer architecture
- Transformer d_model, nhead, num_layers
- Positional/block encodings
- Data preprocessing (tokenization, ASD sampling, etc.)
- Optimizer (AdamW)
- Learning rate schedule

**Result**: Model behavior closely matches zenodo, but optimized for your hardware.

---

## Troubleshooting

### Q: "XPU not available"

**A**: Check XPU installation:

```bash
python -c "import torch; print(torch.xpu.is_available())"
```

Should print `True`.

### Q: "Out of memory" error

**A**: Try reducing batch size to 32:

```yaml
batch_size: 32
```

### Q: "Checkpoint loading failed"

**A**: Make sure checkpoint path is correct:

```bash
ls -lh dingo-T1/02_inference_with_pretrained_models/dingo_t1.pt
```

### Q: "All parameters frozen (trainable=0)"

**A**: Check that `freeze_all_except_last_n_layers()` was called correctly. Look for line in output:

```
Freezing efficiency: 87.5% frozen
```

If you see 100% frozen, something went wrong.

### Q: "Backward pass failed: No gradients"

**A**: Check that last 2 layers are unfrozen. Run test without pre-trained weights:

```bash
python train_xpu_hardware_test.py --config train_settings_xpu_hardware_test.yaml --output_dir ./debug
```

---

## Contact & Support

If you encounter issues:

1. **Check logs**:

   ```bash
   tail -100 xpu_test_results/training_*.log
   ```

2. **Verify environment**:

   ```bash
   python -c "import torch; print(f'XPU: {torch.xpu.is_available()}'); print(f'PyTorch: {torch.__version__}')"
   ```

3. **Review configuration**:
   ```bash
   cat dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml
   ```

---

## Summary

You now have:

- ✅ Freeze/unfreeze methods in transformer.py (130 lines)
- ✅ Optimized XPU config YAML with reduced NSF (70 lines)
- ✅ Hardware test script with XPU init (300+ lines)

**Next action**: Run the hardware test!

```bash
python train_xpu_hardware_test.py \
    --config dingo-T1/01_paper_settings/01_training/03_training/train_settings_xpu_hardware_test.yaml \
    --zenodo_checkpoint dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt \
    --output_dir ./xpu_test_results
```
