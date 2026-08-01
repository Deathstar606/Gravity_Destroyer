# Phase 2 Validation Report

Phase 2 implementation for the 2D conditional Neural Spline Flow is complete. Below are the details corresponding to your validation criteria.

## 1. Modified Files & 2. Reasons for Modification

- `dingo/core/posterior_models/base_model.py`: 
  - **Reason**: Implemented `load_embedding_weights_only` to explicitly filter out the original 15D flow weights during checkpoint loading. The new logic loops over `model_state_dict`, excludes any keys beginning with `flow.`, stores them in a list, prints the excluded keys (so you know what was skipped), and loads the remaining transformer components via `strict=False`.
- `dingo/gw/training/train_pipeline.py`:
  - **Reason**: Updated `initialize_stage` to dynamically read `freeze_all_except_last_n_layers` from the stage configuration instead of hardcoding `n=2`. By shifting this block out of the `if not resume:` statement, we ensure that if a training stage resumes, the correct freezing rules for the transformer are re-applied (as PyTorch checkpoints do not save `requires_grad` flags).
- `dingo/train_settings_phase2.yaml` (New Template):
  - **Reason**: Created a template configuration file demonstrating how to trigger the 2D architecture by explicitly declaring `param_dim: 2`, `context_dim: 134`, `num_flow_steps: 10`, `freeze_all_except_last_n_layers` per stage, and the `inference_parameters` targets `["beta_residual", "chirp_mass"]`.

## 3. Dimensions of Modified Modules

**Posterior Model (`BeyondGRFlowWrapper` / `create_nsf_model`)**
- **Input (Context)**: `(B, 134)` (128D Transformer output + 6 physical/proxy scalars)
- **Input (Flow Variables)**: `(B, 2)` (Specifically: `beta_residual`, `chirp_mass`)
- **Output**: Posterior distribution over `(beta_residual, chirp_mass)` with shape `(B,)` for log-probability evaluations.

## 4. Architecture Confirmations

- **Flow dimension**: `2` (Configured via `param_dim: 2` and targets fetched via `inference_parameters`).
- **Context dimension**: `134` (Built by `BeyondGRFlowWrapper` concatenating the 6 scalars, correctly sized for `context_dim: 134`).
- **10 coupling layers**: Configured via `num_flow_steps: 10`.
- **2 residual blocks**: Configured via `num_transform_blocks: 2`.
- **ELU activation**: Configured via `activation: "elu"`.
- **8 spline bins**: Configured via `num_bins: 8`.
- **Native Dingo permutation**: Confirmed. Dingo's `create_transform` utilizes `transforms.RandomPermutation(features=param_dim)` for mixing. Additionally, `create_base_transform` uses `nflows.utils.create_alternating_binary_mask(param_dim, even=(i % 2 == 0))` to naturally alternate which of the 2 variables is passed through the transform network per layer.

## 5. Checkpoint Loading Confirmation

Confirmed. The new loading logic traverses the `model_state_dict` of the base Dingo-T1 model and explicitly isolates all keys starting with `flow.`. These keys are discarded, and you will see a print statement in the console confirming exactly how many flow parameters were skipped and listing the first 10 skipped keys. All other layers (tokenizer, positional encoding, transformer attention blocks) are loaded correctly.

## 6. Assumptions Made

- **Target Extraction Framework**: Assumed that Dingo's native `SelectStandardizeRepackageParameters` will correctly extract the 2D tensor `y` strictly from the variables listed in `train_settings["data"]["inference_parameters"]`. As long as `inference_parameters` contains exactly `["beta_residual", "chirp_mass"]`, the targets are automatically constructed for the 2D flow.
- **Stage Freezing Configuration**: Assumed that providing `-1` for `freeze_all_except_last_n_layers` corresponds to "unfreeze all layers", `0` corresponds to "freeze all layers", and `2` corresponds to "unfreeze last 2 layers".
- **Architecture Abstraction**: Assumed that modifying the architecture directly via the `train_settings.yaml` (which passes kwargs to `create_nsf_model`) was the preferred method rather than hardcoding the 2D flow sizes directly into `nsf.py`. This preserves the generalized flexibility of Dingo while guaranteeing the 2D Neural Spline Flow is built to specification.
