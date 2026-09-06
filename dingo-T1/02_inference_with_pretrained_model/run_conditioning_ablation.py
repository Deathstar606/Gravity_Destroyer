"""
Conditioning Ablation Diagnostic Script for Beyond-GR Dingo Stage-0 Pipeline.

Compares three conditioner inputs on the SAME diagnostic training data:
  A) EMBEDDING ONLY (128-D)
  B) PARAMETERS ONLY (6-D)
  C) EMBEDDING + PARAMETERS (134-D)

Evaluates:
  - Matched Loss (-log_prob(theta_i, context_i))
  - Shuffled Loss (-log_prob(theta_i, context_j))
  - Delta = Shuffled Loss - Matched Loss
"""

import sys
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, "/home/deathstar/dingorep/dingo")

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.core.nn.nsf import create_nsf_model
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.training.train_builders import set_train_transforms

print("=" * 85)
print("BEYOND-GR CONDITIONING ABLATION DIAGNOSTIC (A vs B vs C)")
print("=" * 85)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
print(f"Using device: {device}")

# -----------------------------------------------------------------------------
# 1. Load Settings and Dataset
# -----------------------------------------------------------------------------
settings_path = "/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml"
t1_ckpt_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"

with open(settings_path, "r") as f:
    train_settings = yaml.safe_load(f)

wfd_path = train_settings["data"]["waveform_dataset_path"]
asd_path = train_settings["training"]["stage_0"]["asd_dataset_path"]

print(f"Loading WaveformDataset from: {wfd_path}")
wfd = WaveformDataset(file_name=wfd_path)

print("Setting up training transforms...")
set_train_transforms(
    wfd=wfd,
    data_settings=train_settings["data"],
    asd_dataset_path=asd_path,
    print_output=False,
)

# -----------------------------------------------------------------------------
# 2. Load Pretrained Embedding Network
# -----------------------------------------------------------------------------
print(f"Loading pretrained embedding network from: {t1_ckpt_path}")
pm_base = build_model_from_kwargs(
    filename=t1_ckpt_path,
    pretraining=False,
    pretrained_embedding_net=None,
    load_embedding_only=True,
    device=device,
    print_output=False,
)

embedding_net = pm_base.network.embedding_net.to(device)
embedding_net.eval()
for param in embedding_net.parameters():
    param.requires_grad = False

print("Pretrained embedding network frozen in eval mode.")

# -----------------------------------------------------------------------------
# 3. Precompute Diagnostic Batches (Fixed Dataset Subset)
# -----------------------------------------------------------------------------
batch_size = 64
num_train_batches = 40
num_val_batches = 15

total_needed = (num_train_batches + num_val_batches) * batch_size
print(f"\nGenerating and precomputing {total_needed} samples across {num_train_batches + num_val_batches} batches...")

loader = DataLoader(wfd, batch_size=batch_size, shuffle=False, num_workers=0)

all_batches = []
batch_iter = iter(loader)

with torch.no_grad():
    for b_idx in range(num_train_batches + num_val_batches):
        batch = next(batch_iter)
        # batch structure:
        # selected_keys = ['inference_parameters', 'waveform', 'context_parameters', 'position', 'drop_token_mask']
        theta = batch[0].to(device).float()              # [B, 2] -> [beta_residual, chirp_mass]
        waveform = batch[1].to(device).float()           # [B, N, 48]
        context_params = batch[2].to(device).float()     # [B, 6] -> [beta_proxy, q, chi1z, chi2z, dL, theta_jn]
        position = batch[3].to(device).float()           # [B, N, 3]
        padding_mask = batch[4].to(device).bool()        # [B, N]

        embed = embedding_net(waveform, position, padding_mask) # [B, 128]
        if isinstance(embed, tuple):
            embed = embed[0]

        context_both = torch.cat([embed, context_params], dim=-1) # [B, 134]

        all_batches.append({
            "theta": theta,
            "context_A": embed,                   # 128-D
            "context_B": context_params,           # 6-D
            "context_C": context_both,             # 134-D
        })

train_batches = all_batches[:num_train_batches]
val_batches = all_batches[num_train_batches:]

print(f"Precomputed {len(train_batches)} training batches and {len(val_batches)} validation batches.")
print(f"Target theta shape: {train_batches[0]['theta'].shape}")
print(f"Conditioner A shape (Embedding only): {train_batches[0]['context_A'].shape}")
print(f"Conditioner B shape (Params only):    {train_batches[0]['context_B'].shape}")
print(f"Conditioner C shape (Emb + Params):   {train_batches[0]['context_C'].shape}")

# -----------------------------------------------------------------------------
# 4. Define Model Builder and Training Routine
# -----------------------------------------------------------------------------
num_flow_steps = 8
base_transform_kwargs = {
    "hidden_dim": 256,
    "num_transform_blocks": 2,
    "activation": "elu",
    "batch_norm": False,
    "layer_norm": True,
    "dropout_probability": 0.0,
    "num_bins": 8,
    "base_transform_type": "rq-coupling",
    "context_in_initial_layer": True,
}

def build_diagnostic_flow(context_dim: int, seed: int = 42):
    torch.manual_seed(seed)
    flow = create_nsf_model(
        input_dim=2,
        context_dim=context_dim,
        num_flow_steps=num_flow_steps,
        base_transform_kwargs=base_transform_kwargs,
    ).to(device)
    return flow

def train_diagnostic_model(conditioner_name: str, context_key: str, context_dim: int, num_epochs: int = 5, lr: float = 1e-3):
    print("\n" + "=" * 80)
    print(f"TRAINING MODEL {conditioner_name} (Context Dim = {context_dim})")
    print("=" * 80)

    flow = build_diagnostic_flow(context_dim=context_dim, seed=42)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)

    # Initial layer shape check
    for name, param in flow.named_parameters():
        if "initial_layer.weight" in name:
            print(f"[{conditioner_name}] First initial_layer.weight shape: {list(param.shape)}")
            break

    flow.train()
    step = 0
    final_train_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_losses = []
        for b in train_batches:
            theta = b["theta"]
            context = b[context_key]

            optimizer.zero_grad()
            log_prob = flow.log_prob(theta, context)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            step += 1

        mean_loss = np.mean(epoch_losses)
        final_train_loss = mean_loss
        print(f"  Epoch {epoch:2d}/{num_epochs:2d} | Step {step:4d} | Train NLL Loss: {mean_loss:.4f}")

    # Evaluation on Validation Batches
    flow.eval()
    matched_losses = []
    shuffled_losses = []

    with torch.no_grad():
        for b in val_batches:
            theta = b["theta"]
            context = b[context_key]
            B = theta.shape[0]

            # 1. Matched Loss
            lp_matched = flow.log_prob(theta, context)
            matched_losses.append((-lp_matched).mean().item())

            # 2. Shuffled Loss (cyclically roll context by 1 so no pair is matched)
            perm = torch.roll(torch.arange(B, device=device), shifts=1)
            shuffled_context = context[perm]
            lp_shuffled = flow.log_prob(theta, shuffled_context)
            shuffled_losses.append((-lp_shuffled).mean().item())

    val_matched_mean = np.mean(matched_losses)
    val_matched_std = np.std(matched_losses)
    val_shuffled_mean = np.mean(shuffled_losses)
    val_shuffled_std = np.std(shuffled_losses)
    delta = val_shuffled_mean - val_matched_mean

    print(f"\n[{conditioner_name}] Validation Results:")
    print(f"  Matched Loss  : {val_matched_mean:.6f} +/- {val_matched_std:.6f}")
    print(f"  Shuffled Loss : {val_shuffled_mean:.6f} +/- {val_shuffled_std:.6f}")
    print(f"  Delta (Shuffled - Matched): {delta:+.6f}")

    return {
        "name": conditioner_name,
        "context_dim": context_dim,
        "train_loss": final_train_loss,
        "matched_loss": val_matched_mean,
        "shuffled_loss": val_shuffled_mean,
        "delta": delta,
    }

# -----------------------------------------------------------------------------
# 5. Run Ablation for A, B, and C
# -----------------------------------------------------------------------------
results = {}

# A) EMBEDDING ONLY
results["A"] = train_diagnostic_model(
    conditioner_name="A) EMBEDDING ONLY",
    context_key="context_A",
    context_dim=128,
    num_epochs=8,
    lr=1e-3,
)

# B) PARAMETERS ONLY
results["B"] = train_diagnostic_model(
    conditioner_name="B) PARAMETERS ONLY",
    context_key="context_B",
    context_dim=6,
    num_epochs=8,
    lr=1e-3,
)

# C) EMBEDDING + PARAMETERS
results["C"] = train_diagnostic_model(
    conditioner_name="C) EMBEDDING + PARAMETERS",
    context_key="context_C",
    context_dim=134,
    num_epochs=8,
    lr=1e-3,
)

# -----------------------------------------------------------------------------
# 6. Final Comparison Report & Interpretation
# -----------------------------------------------------------------------------
print("\n" + "=" * 95)
print("FINAL CONDITIONING ABLATION COMPARISON")
print("=" * 95)
print(f"{'Conditioner Mode':<30} | {'Dim':<5} | {'Train Loss':<12} | {'Matched Loss':<14} | {'Shuffled Loss':<14} | {'Delta (Δ)':<12}")
print("-" * 95)

for key in ["A", "B", "C"]:
    r = results[key]
    print(f"{r['name']:<30} | {r['context_dim']:<5} | {r['train_loss']:<12.4f} | {r['matched_loss']:<14.6f} | {r['shuffled_loss']:<14.6f} | {r['delta']:<+12.6f}")

print("=" * 95)

print("\n--- INTERPRETATION & FINDINGS ---")
delta_A = results["A"]["delta"]
delta_B = results["B"]["delta"]
delta_C = results["C"]["delta"]

print(f"1. Embedding Only (128-D) Δ      : {delta_A:+.6f}")
if delta_A > 0.05:
    print("   -> The 128-D transformer embedding carries STRONG conditional information about theta.")
else:
    print("   -> The 128-D transformer embedding carries minimal/weak conditional information about theta.")

print(f"2. Parameters Only (6-D) Δ       : {delta_B:+.6f}")
if delta_B > 0.05:
    print("   -> The 6 explicit parameters alone carry STRONG conditional information about theta.")
else:
    print("   -> The 6 explicit parameters alone carry minimal/weak conditional information about theta.")

print(f"3. Embedding + Parameters (134-D) Δ: {delta_C:+.6f}")
if delta_C > max(delta_A, delta_B):
    print("   -> Combining Embedding + Parameters yields HIGHER conditional constraint than either individually.")
else:
    print(f"   -> Combining provides comparable constraint (Delta = {delta_C:+.6f}).")

print("\n[STOPPING DIAGNOSTICS IMMEDIATELY — NO FULL TRAINING INITIATED]")
