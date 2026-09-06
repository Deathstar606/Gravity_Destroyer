"""
SMALL CONDITIONAL NSF DIAGNOSTIC EXPERIMENT
============================================
Objective:
Determine whether a freshly initialized 2D conditional NSF with a drastically
smaller conditioner (e.g. 4 flow steps, 1 residual block, hidden width 64)
can learn conditional dependence p(theta | context) rather than defaulting
to the marginal p(theta).

Tests:
  1. Training on fixed real batch with Matched vs Shuffled NLL evaluation
  2. Detailed Parameter Update Audit (context vs spline vs remaining)
  3. Context Ablation (Matched, Shuffled, Zero, Random, Emb-only, Params-only)
  4. Context Normalization & Usability Statistics
  5. Quantitative Final Report and Verdict (PASS / PARTIAL / FAIL)
"""

import sys
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/home/deathstar/dingorep/dingo")

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.core.utils.torchutils import print_number_of_model_parameters
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.training.train_builders import set_train_transforms
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Run small conditional NSF diagnostic")
    parser.add_argument("--hidden_dim", type=int, default=64, choices=[64, 128, 256, 512],
                        help="Hidden dimension of conditioner residual blocks (default: 64)")
    parser.add_argument("--num_flow_steps", type=int, default=4,
                        help="Number of flow coupling steps (default: 4)")
    parser.add_argument("--num_transform_blocks", type=int, default=1,
                        help="Number of residual blocks per transform net (default: 1)")
    parser.add_argument("--num_bins", type=int, default=8,
                        help="Number of spline bins (default: 8)")
    parser.add_argument("--activation", type=str, default="elu",
                        help="Activation function for conditioner (default: elu)")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of optimizer steps on fixed batch (default: 500)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AdamW optimizer (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.005,
                        help="Weight decay for AdamW (default: 0.005)")
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device (default: xpu if available, else cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()

def generate_derangement(n, device):
    """
    Generate a random permutation of length n with NO fixed points (p[i] != i for all i).
    This guarantees strictly non-identical context pairing for every element in the batch.
    """
    perm = torch.randperm(n, device=device)
    # Fix any fixed points by swapping with neighbor
    fixed = (perm == torch.arange(n, device=device))
    while fixed.any():
        for i in torch.where(fixed)[0]:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
        fixed = (perm == torch.arange(n, device=device))
    return perm

def categorize_parameters(flow_net):
    """
    Categorize all trainable parameters in the flow into:
      1. context_proj: Context projection / conditioning layers
         (e.g., initial_layer, context_layer in MyResidualBlock)
      2. spline_net: Spline-producing layers (residual block linear layers, final resize layers)
      3. remaining_flow: Linear / LU transforms, permutations, base distribution
    """
    param_groups = {
        "context_proj": {},
        "spline_net": {},
        "remaining_flow": {}
    }
    
    for name, param in flow_net.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter belongs to context projection
        is_context = False
        if "context_layer" in name:
            is_context = True
        elif "initial_layer" in name:
            is_context = True
            
        if is_context:
            param_groups["context_proj"][name] = param
        elif "transform_net" in name or "blocks" in name or "resize_layers" in name:
            param_groups["spline_net"][name] = param
        else:
            param_groups["remaining_flow"][name] = param
            
    return param_groups

def compute_relative_updates(initial_params, current_params, epsilon=1e-8):
    """
    Calculate: ||parameter_after - parameter_before|| / (||parameter_before|| + epsilon)
    Aggregated across a group of parameters.
    """
    if not initial_params:
        return 0.0
    total_delta_sq = 0.0
    total_initial_sq = 0.0
    
    for name, p_init in initial_params.items():
        p_curr = current_params[name]
        delta = p_curr - p_init
        total_delta_sq += (delta.norm() ** 2).item()
        total_initial_sq += (p_init.norm() ** 2).item()
        
    delta_norm = np.sqrt(total_delta_sq)
    initial_norm = np.sqrt(total_initial_sq)
    return delta_norm / (initial_norm + epsilon)

def compute_grad_norm(params, epsilon=1e-8):
    """
    Compute total L2 gradient norm across a dictionary of parameters.
    """
    if not params:
        return 0.0
    total_grad_sq = 0.0
    for name, p in params.items():
        if p.grad is not None:
            total_grad_sq += (p.grad.norm() ** 2).item()
    return np.sqrt(total_grad_sq)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("=" * 90)
    print("SMALL CONDITIONAL NSF DIAGNOSTIC EXPERIMENT")
    print("=" * 90)
    
    # -------------------------------------------------------------------------
    # 1. Device Selection
    # -------------------------------------------------------------------------
    if args.device is not None:
        device = args.device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    # -------------------------------------------------------------------------
    # 2. Load Pretrained Embedding & Build Small 2D NSF Model
    # -------------------------------------------------------------------------
    model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
    settings_path = "/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml"
    
    print(f"\nLoading pretrained embedding from: {model_path}")
    embedding_model = build_model_from_kwargs(
        filename=model_path,
        pretraining=False,
        pretrained_embedding_net=None,
        load_embedding_only=True,
        device=device,
        print_output=False,
    )
    
    with open(settings_path, "r") as f:
        train_settings = yaml.safe_load(f)
        
    model_settings = copy.deepcopy(train_settings)
    
    # Configure Small 2D NSF
    model_settings["model"]["posterior_kwargs"]["input_dim"] = 2
    model_settings["model"]["posterior_kwargs"]["context_dim"] = 134
    model_settings["model"]["posterior_kwargs"]["num_flow_steps"] = args.num_flow_steps
    model_settings["model"]["posterior_kwargs"]["base_transform_kwargs"] = {
        "hidden_dim": args.hidden_dim,
        "num_transform_blocks": args.num_transform_blocks,
        "activation": args.activation,
        "batch_norm": False,
        "layer_norm": True,
        "dropout_probability": 0.0,
        "num_bins": args.num_bins,
        "base_transform_type": "rq-coupling",
        "context_in_initial_layer": True,
    }
    
    # Configure Embedding Wrapper
    model_settings["model"]["embedding_kwargs"]["tokenizer_kwargs"].update({
        "input_dims": [207, 48],
        "output_dim": 1024,
        "num_blocks": 3,
        "context_features": 5,
    })
    model_settings["model"]["embedding_kwargs"]["final_net_kwargs"] = {
        "input_dim": 1024,
        "output_dim": 128,
        "activation": "elu",
    }
    model_settings["model"]["embedding_kwargs"]["added_context"] = False
    
    dataset_settings = {
        "domain_dict": {
            "type": "MultibandedFrequencyDomain",
            "base_domain_kwargs": {"f_min": 20.0, "f_max": 2048.0, "delta_f": 0.125},
            "bands": [(20.0, 128.0, 1), (128.0, 2048.0, 4)],
        },
    }
    
    print("\nBuilding small 2D NSF model...")
    model_2d = build_model_from_kwargs(
        settings={"dataset_settings": dataset_settings, "train_settings": model_settings},
        pretraining=False,
        pretrained_embedding_net=embedding_model.network.embedding_net,
        device=device,
        print_output=False,
    )
    # Load pretrained embedding weights into new model
    model_2d.load_embedding_weights_only(model_path, device=device)
    model = model_2d
    
    flow_net = model.network.flow
    embedding_net = model.network.embedding_net
    
    # Freeze embedding network (Stage 0 setup)
    embedding_net.freeze_all_except_last_n_layers(n=0)
    embedding_net.eval()
    for p in embedding_net.parameters():
        p.requires_grad = False
        
    print("\nModel Architecture Summary:")
    print(f"  Target dimension : 2 (beta_residual, chirp_mass)")
    print(f"  Context dimension: 134 (128-D embedding + 6 explicit params)")
    print(f"  Flow steps       : {args.num_flow_steps}")
    print(f"  Blocks per step  : {args.num_transform_blocks}")
    print(f"  Hidden width     : {args.hidden_dim}")
    print(f"  Spline bins      : {args.num_bins}")
    print(f"  Activation       : {args.activation}")
    print(f"  Context in initial: True")
    print_number_of_model_parameters(model.network)
    
    # -------------------------------------------------------------------------
    # 3. Load 1 Real Training Batch
    # -------------------------------------------------------------------------
    wfd_path = train_settings["data"]["waveform_dataset_path"]
    asd_path = train_settings["training"]["stage_0"]["asd_dataset_path"]
    
    print(f"\nLoading WaveformDataset from: {wfd_path}")
    wfd = WaveformDataset(file_name=wfd_path)
    set_train_transforms(
        wfd=wfd,
        data_settings=train_settings["data"],
        asd_dataset_path=asd_path,
        print_output=False,
    )
    
    loader = DataLoader(wfd, batch_size=256, shuffle=False, num_workers=0)
    raw_batch = next(iter(loader))
    
    theta_batch = raw_batch[0].to(device).float()        # [B, 2] = [beta_residual, chirp_mass]
    wf_batch = raw_batch[1].to(device).float()           # [B, N, 48]
    ctx_params_batch = raw_batch[2].to(device).float()   # [B, 6] = [beta_proxy, q, chi1z, chi2z, dL, theta_jn]
    pos_batch = raw_batch[3].to(device).float()          # [B, N, 3]
    mask_batch = raw_batch[4].to(device).bool()          # [B, N]
    
    B = theta_batch.shape[0]
    print(f"\nExtracted 1 Real Training Batch: B = {B}")
    
    # Compute 128-D transformer embeddings once (since embedding is frozen)
    with torch.no_grad():
        emb_batch = embedding_net(wf_batch, pos_batch, mask_batch)
        if isinstance(emb_batch, tuple):
            emb_batch = emb_batch[0]
            
    # Form the 134-D context: [128-D embedding || 6-D explicit parameters]
    context_batch = torch.cat([emb_batch, ctx_params_batch], dim=-1)
    
    # -------------------------------------------------------------------------
    # 4. Context Normalization & Usability Audit
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CONTEXT NORMALIZATION & NUMERICAL USABILITY AUDIT")
    print("=" * 90)
    emb_norms = emb_batch.norm(dim=-1)
    ctx_norms = context_batch.norm(dim=-1)
    
    param_names = ["beta_proxy", "mass_ratio", "chi1z", "chi2z", "luminosity_distance", "theta_jn"]
    print(f"Embedding (128-D): mean={emb_batch.mean().item():.4f}, std={emb_batch.std().item():.4f}, "
          f"min={emb_batch.min().item():.4f}, max={emb_batch.max().item():.4f}, mean_norm={emb_norms.mean().item():.4f}")
    
    print("\nSix Explicit Parameters Statistics:")
    for idx, pname in enumerate(param_names):
        vals = ctx_params_batch[:, idx]
        print(f"  [{idx}] {pname:<22}: mean={vals.mean().item():>10.4f}, std={vals.std().item():>10.4f}, "
              f"min={vals.min().item():>10.4f}, max={vals.max().item():>10.4f}")
              
    print(f"\nTarget Theta Statistics:")
    print(f"  [0] beta_residual: mean={theta_batch[:, 0].mean().item():.4f}, std={theta_batch[:, 0].std().item():.4f}, "
          f"min={theta_batch[:, 0].min().item():.4f}, max={theta_batch[:, 0].max().item():.4f}")
    print(f"  [1] chirp_mass   : mean={theta_batch[:, 1].mean().item():.4f}, std={theta_batch[:, 1].std().item():.4f}, "
          f"min={theta_batch[:, 1].min().item():.4f}, max={theta_batch[:, 1].max().item():.4f}")
          
    has_nan_or_inf = torch.isnan(context_batch).any() or torch.isinf(context_batch).any()
    print(f"\nNumerical usability check: NaN or Inf present? -> {has_nan_or_inf.item()}")
    assert not has_nan_or_inf, "Pathological context detected with NaN or Inf values!"
    print("Context vectors are numerically finite and valid.")
    
    # -------------------------------------------------------------------------
    # 5. Parameter Update Audit Setup
    # -------------------------------------------------------------------------
    param_groups = categorize_parameters(flow_net)
    print("\nParameter Categorization:")
    print(f"  Context projection layers : {len(param_groups['context_proj'])} tensors")
    for name, p in param_groups["context_proj"].items():
        print(f"    - {name}: shape={list(p.shape)}")
    print(f"  Spline-producing layers   : {len(param_groups['spline_net'])} tensors")
    print(f"  Remaining flow layers     : {len(param_groups['remaining_flow'])} tensors")
    
    # Save initial snapshot of all flow weights
    initial_weights = {
        name: p.clone().detach()
        for name, p in flow_net.named_parameters()
        if p.requires_grad
    }
    
    initial_groups = {
        group_name: {name: p.clone().detach() for name, p in group_dict.items()}
        for group_name, group_dict in param_groups.items()
    }
    
    # Setup AdamW optimizer matching Stage-0 settings
    opt_cfg = train_settings["training"]["stage_0"]["optimizer"]
    optimizer = torch.optim.AdamW(
        flow_net.parameters(),
        lr=args.lr,
        betas=tuple(opt_cfg.get("betas", [0.8, 0.99])),
        weight_decay=args.weight_decay,
    )
    
    # Fixed derangement permutation for shuffled evaluation
    # Using fixed derangement ensures every theta_i is paired with context_j where j != i
    shuffled_perm = generate_derangement(B, device)
    
    # Reporting checkpoints
    checkpoints = sorted(list(set([0, 1, 25, 50, 100, 200, 300, args.num_steps])))
    history = []
    
    # -------------------------------------------------------------------------
    # 6. Evaluation Function
    # -------------------------------------------------------------------------
    def evaluate(current_step):
        flow_net.eval()
        with torch.no_grad():
            # A. Matched: theta_i + context_i
            lp_matched = flow_net.log_prob(theta_batch, context_batch)
            nll_matched = -lp_matched.mean().item()
            
            # B. Shuffled: theta_i + context_j
            lp_shuffled = flow_net.log_prob(theta_batch, context_batch[shuffled_perm])
            nll_shuffled = -lp_shuffled.mean().item()
            
            delta = nll_shuffled - nll_matched
            
            # C. Zero context
            zero_context = torch.zeros_like(context_batch)
            lp_zero = flow_net.log_prob(theta_batch, zero_context)
            nll_zero = -lp_zero.mean().item()
            
            # D. Random context (standard normal)
            torch.manual_seed(9999 + current_step)
            rand_context = torch.randn_like(context_batch)
            lp_rand = flow_net.log_prob(theta_batch, rand_context)
            nll_rand = -lp_rand.mean().item()
            
            # E. Embedding only (parameters zeroed)
            emb_only_ctx = context_batch.clone()
            emb_only_ctx[:, 128:] = 0.0
            lp_emb = flow_net.log_prob(theta_batch, emb_only_ctx)
            nll_emb_only = -lp_emb.mean().item()
            
            # F. Parameters only (embedding zeroed)
            params_only_ctx = context_batch.clone()
            params_only_ctx[:, :128] = 0.0
            lp_params = flow_net.log_prob(theta_batch, params_only_ctx)
            nll_params_only = -lp_params.mean().item()
            
        current_weights = {
            name: p.detach()
            for name, p in flow_net.named_parameters()
            if p.requires_grad
        }
        current_grouped = {
            group_name: {name: p.detach() for name, p in group_dict.items()}
            for group_name, group_dict in param_groups.items()
        }
        
        rel_upd_ctx = compute_relative_updates(initial_groups["context_proj"], current_grouped["context_proj"])
        rel_upd_spline = compute_relative_updates(initial_groups["spline_net"], current_grouped["spline_net"])
        rel_upd_rem = compute_relative_updates(initial_groups["remaining_flow"], current_grouped["remaining_flow"])
        
        return {
            "step": current_step,
            "matched_nll": nll_matched,
            "shuffled_nll": nll_shuffled,
            "delta": delta,
            "zero_nll": nll_zero,
            "rand_nll": nll_rand,
            "emb_only_nll": nll_emb_only,
            "params_only_nll": nll_params_only,
            "rel_upd_ctx": rel_upd_ctx,
            "rel_upd_spline": rel_upd_spline,
            "rel_upd_rem": rel_upd_rem,
        }
        
    print("\n" + "=" * 90)
    print("TEST 1: SMALL CONDITIONAL NSF TRAINING (MATCHED vs SHUFFLED NLL)")
    print("=" * 90)
    print(f"{'Step':>6} | {'Matched NLL':>12} | {'Shuffled NLL':>13} | {'Delta (Shuf-Mat)':>17} | "
          f"{'|Grad(Ctx)|':>12} | {'Upd(Ctx)':>10} | {'Upd(Spline)':>11} | {'Upd(Flow)':>10}")
    print("-" * 105)
    
    # Initial evaluation at Step 0
    init_eval = evaluate(0)
    init_eval["grad_ctx"] = 0.0
    init_eval["grad_spline"] = 0.0
    init_eval["grad_rem"] = 0.0
    history.append(init_eval)
    
    print(f"{0:>6d} | {init_eval['matched_nll']:>12.6f} | {init_eval['shuffled_nll']:>13.6f} | "
          f"{init_eval['delta']:>+17.6f} | {'---':>12} | {init_eval['rel_upd_ctx']:>10.6f} | "
          f"{init_eval['rel_upd_spline']:>11.6f} | {init_eval['rel_upd_rem']:>10.6f}")
          
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for step in range(1, args.num_steps + 1):
        flow_net.train()
        optimizer.zero_grad()
        
        # Training loss: NLL of matched pairs -log_prob(theta_i, context_i)
        lp = flow_net.log_prob(theta_batch, context_batch)
        loss = -lp.mean()
        loss.backward()
        
        # Compute gradient norms before optimizer step
        grad_ctx = compute_grad_norm(param_groups["context_proj"])
        grad_spline = compute_grad_norm(param_groups["spline_net"])
        grad_rem = compute_grad_norm(param_groups["remaining_flow"])
        
        optimizer.step()
        
        if step in checkpoints:
            eval_res = evaluate(step)
            eval_res["grad_ctx"] = grad_ctx
            eval_res["grad_spline"] = grad_spline
            eval_res["grad_rem"] = grad_rem
            history.append(eval_res)
            
            print(f"{step:>6d} | {eval_res['matched_nll']:>12.6f} | {eval_res['shuffled_nll']:>13.6f} | "
                  f"{eval_res['delta']:>+17.6f} | {grad_ctx:>12.6e} | {eval_res['rel_upd_ctx']:>10.6f} | "
                  f"{eval_res['rel_upd_spline']:>11.6f} | {eval_res['rel_upd_rem']:>10.6f}")
                  
    # -------------------------------------------------------------------------
    # TEST 2: PARAMETER UPDATE AUDIT DETAILS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 2: DETAILED PARAMETER UPDATE AUDIT")
    print("=" * 90)
    final_eval = history[-1]
    
    print("Relative updates from initialization to final step:")
    print(f"  Context projection weights (initial_layer + context_layer): {final_eval['rel_upd_ctx']:.6f}")
    print(f"  Spline-producing network weights (blocks + resize_layers) : {final_eval['rel_upd_spline']:.6f}")
    print(f"  Remaining flow weights (LU linear transforms, permutations): {final_eval['rel_upd_rem']:.6f}")
    
    print("\nContext Projection Layers Breakdown:")
    for name, p in param_groups["context_proj"].items():
        delta_p = p.detach() - initial_weights[name]
        rel_upd = (delta_p.norm() / (initial_weights[name].norm() + 1e-8)).item()
        grad_val = p.grad.norm().item() if p.grad is not None else 0.0
        print(f"  {name:<55} | Rel Update: {rel_upd:.6f} | Grad Norm: {grad_val:.6e}")
        
    # Categorize behavior
    if final_eval["grad_ctx"] < 1e-6:
        grad_verdict = "B. Receiving extremely small gradients"
    elif final_eval["rel_upd_ctx"] < 1e-4:
        grad_verdict = "A. Receiving gradients but not updating"
    elif final_eval["rel_upd_ctx"] > 0.01:
        grad_verdict = "C. Updating normally with substantial magnitude"
    else:
        grad_verdict = "D. Updating slowly, potentially overshadowed"
    print(f"\nConditioner Parameter Update Classification: {grad_verdict}")
    
    # -------------------------------------------------------------------------
    # TEST 3: CONTEXT ABLATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST 3: CONTEXT ABLATION (AT FINAL STEP)")
    print("=" * 90)
    matched_nll = final_eval["matched_nll"]
    
    print(f"{'Context Type':<30} | {'Mean NLL':>12} | {'Diff vs Matched':>17} | {'Interpretation':<25}")
    print("-" * 90)
    print(f"{'A. Matched Context':<30} | {matched_nll:>12.6f} | {'0.000000 (ref)':>17} | {'Baseline':<25}")
    print(f"{'B. Shuffled Context':<30} | {final_eval['shuffled_nll']:>12.6f} | "
          f"{final_eval['shuffled_nll'] - matched_nll:>+17.6f} | "
          f"{'Target/Context mismatch':<25}")
    print(f"{'C. Zero Context':<30} | {final_eval['zero_nll']:>12.6f} | "
          f"{final_eval['zero_nll'] - matched_nll:>+17.6f} | "
          f"{'Conditioning removed':<25}")
    print(f"{'D. Random Context':<30} | {final_eval['rand_nll']:>12.6f} | "
          f"{final_eval['rand_nll'] - matched_nll:>+17.6f} | "
          f"{'Uncorrelated noise':<25}")
    print(f"{'E. Embedding-Only Context':<30} | {final_eval['emb_only_nll']:>12.6f} | "
          f"{final_eval['emb_only_nll'] - matched_nll:>+17.6f} | "
          f"{'Explicit params zeroed':<25}")
    print(f"{'F. Parameters-Only Context':<30} | {final_eval['params_only_nll']:>12.6f} | "
          f"{final_eval['params_only_nll'] - matched_nll:>+17.6f} | "
          f"{'Waveform emb zeroed':<25}")
          
    # -------------------------------------------------------------------------
    # REQUIRED FINAL REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("REQUIRED FINAL REPORT")
    print("=" * 90)
    
    init_matched = history[0]["matched_nll"]
    init_shuffled = history[0]["shuffled_nll"]
    init_delta = history[0]["delta"]
    final_matched = final_eval["matched_nll"]
    final_shuffled = final_eval["shuffled_nll"]
    final_delta = final_eval["delta"]
    
    print(f"Model architecture:")
    print(f"  Target dim: 2, Context dim: 134, Steps: {args.num_flow_steps}, Blocks: {args.num_transform_blocks}, Hidden width: {args.hidden_dim}")
    print(f"Context normalization statistics:")
    print(f"  Embedding mean: {emb_batch.mean().item():.4f}, std: {emb_batch.std().item():.4f}, norm: {emb_norms.mean().item():.4f}")
    print(f"  Params mean   : {ctx_params_batch.mean().item():.4f}, std: {ctx_params_batch.std().item():.4f}")
    print(f"Initial matched NLL  : {init_matched:.6f}")
    print(f"Initial shuffled NLL : {init_shuffled:.6f}")
    print(f"Final matched NLL    : {final_matched:.6f} (Change: {final_matched - init_matched:+.6f})")
    print(f"Final shuffled NLL   : {final_shuffled:.6f} (Change: {final_shuffled - init_shuffled:+.6f})")
    print(f"Initial Delta        : {init_delta:+.6f}")
    print(f"Final Delta          : {final_delta:+.6f}")
    
    print("\nDelta progression over optimizer steps:")
    for h in history:
        print(f"  Step {h['step']:>4d}: Matched NLL={h['matched_nll']:>9.4f}, "
              f"Shuffled NLL={h['shuffled_nll']:>9.4f}, Delta={h['delta']:>+9.4f}")
              
    print(f"\nConditioner gradient norms          : {final_eval['grad_ctx']:.6e}")
    print(f"Conditioner relative parameter updates: {final_eval['rel_upd_ctx']:.6f}")
    print(f"Spline-net relative updates          : {final_eval['rel_upd_spline']:.6f}")
    print(f"Flow/non-conditioner relative updates : {final_eval['rel_upd_rem']:.6f}")
    print(f"Zero-context result (Delta vs match) : {final_eval['zero_nll'] - final_matched:+.6f}")
    print(f"Random-context result (Delta vs match): {final_eval['rand_nll'] - final_matched:+.6f}")
    print(f"Embedding-only result (Delta vs match): {final_eval['emb_only_nll'] - final_matched:+.6f}")
    print(f"Parameter-only result (Delta vs match): {final_eval['params_only_nll'] - final_matched:+.6f}")
    
    # -------------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------------
    print("\n" + "-" * 90)
    print("FINAL VERDICT")
    print("-" * 90)
    
    # Criteria:
    # PASS: Final Delta > 0.5 (or matched NLL improves significantly faster than shuffled, with Delta > 0.3)
    # PARTIAL: Final Delta between 0.05 and 0.5, or context updates occur but separation is modest
    # FAIL: Final Delta < 0.05 (shuffled and matched losses remain virtually identical)
    if final_delta >= 0.5:
        verdict = "PASS"
        verdict_expl = (
            f"The small conditional flow successfully learns context dependence (Delta = {final_delta:+.4f} > 0.5). "
            f"Matched pairs achieve substantially lower NLL than shuffled pairs. "
            f"Conditioner parameters update normally (relative update = {final_eval['rel_upd_ctx']:.4f}). "
            f"This demonstrates that conditional information in the 134-D context is readily accessible "
            f"when the conditioner capacity is scaled appropriately."
        )
    elif final_delta >= 0.05:
        verdict = "PARTIAL"
        verdict_expl = (
            f"Conditional dependence exists but is weak/unstable (Delta = {final_delta:+.4f}, between 0.05 and 0.5). "
            f"While the model differentiates matched from shuffled context, the gap is modest. "
            f"The smaller conditioner helps, but optimization dynamics or feature representation "
            f"still leave substantial room for marginal domination."
        )
    else:
        verdict = "FAIL"
        verdict_expl = (
            f"The small conditional flow still behaves approximately unconditional (Delta = {final_delta:+.4f} < 0.05). "
            f"Matched and shuffled NLL remain virtually identical despite optimization. "
            f"The failure to separate indicates that the problem is not merely excessive conditioner capacity."
        )
        
    print(f"VERDICT: {verdict} — {verdict_expl}")
    
    print("\nStage-0 Architecture & Training Formulation Implications:")
    if verdict == "PASS":
        print("  -> SUPPORTS changing the main Stage-0 architecture: reducing the conditioner size, "
              "lowering coupling layers from 10 to 4-6, and reducing hidden width from 512 to 64-128 "
              "will prevent the model from finding lazy marginal solutions and force strong context dependence.")
    elif verdict == "PARTIAL":
        print("  -> SUPPORTS architectural modifications in combination with training formulation adjustments: "
              "reducing conditioner capacity helps, but must be paired with explicit contrastive regularization, "
              "feature reweighting (e.g. scaling explicit parameters), or an adjusted learning rate ratio.")
    else:
        print("  -> DOES NOT support capacity reduction alone as the fix: the lack of conditional learning "
              "stems from data representation, target formulation (beta_residual vs beta_proxy alignment), "
              "or loss landscape rather than conditioner over-parameterization.")
              
    print("=" * 90)
    print("DIAGNOSTIC EXPERIMENT COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    main()
