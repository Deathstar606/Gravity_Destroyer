"""
CONTROLLED BEYOND-GR FLOW DIAGNOSTIC SUITE
============================================
Three Controlled Tests with the Originally Configured NSF:
  Test 1: q(beta0 | embedding)                       [1D flow, 128D context]
  Test 2: q(beta0, chirp_mass | embedding)            [2D flow, 128D context]
  Test 3: q(beta0, chirp_mass | embedding, 5 params) [2D flow, 133D context]

Key Principles:
- Uses the originally configured NSF (10 flow steps, 2 blocks, hidden width 512, 8 bins).
- Target is beta_true / beta0 directly (range [-5, +5]), NOT beta_residual.
- NO SampleBeyondGRProxy, NO OnlineBeyondGRRotation.
- NO beta_proxy anywhere in target or conditioning.
- Fixed real batch: 64 samples for 500 optimizer steps.
- Parameter update audit: context projection vs spline net vs flow.
- Gradient audits and target localization (beta0 vs chirp_mass).
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
from dingo.core.nn.nsf import create_nsf_model
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.training.train_builders import set_train_transforms
from dingo.gw.transforms.beyond_gr_transforms import OnlineBeyondGRRotation, SampleBeyondGRProxy
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Beyond-GR Controlled Diagnostic Suite")
    parser.add_argument("--test", type=str, default="all", choices=["1", "2", "3", "all"],
                        help="Which test to run: 1, 2, 3, or all (default: all)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--num_steps", type=int, default=500,
                        help="Number of optimizer steps (default: 500)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for AdamW (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.005,
                        help="Weight decay for AdamW (default: 0.005)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: xpu, cuda, cpu (default: auto-detect)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # Flow architecture options (default to original configuration)
    parser.add_argument("--num_flow_steps", type=int, default=10,
                        help="Number of flow coupling steps (default: 10)")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension of residual blocks (default: 512)")
    parser.add_argument("--num_transform_blocks", type=int, default=2,
                        help="Residual blocks per coupling net (default: 2)")
    parser.add_argument("--num_bins", type=int, default=8,
                        help="Number of spline bins (default: 8)")
    return parser.parse_args()

def generate_derangement(n, device):
    """
    Generate a random permutation with NO fixed points (p[i] != i for all i).
    Ensures every target is paired with a strictly different context sample.
    """
    perm = torch.randperm(n, device=device)
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
      1. context_proj: Context projection / conditioning layers (initial_layer, context_layer)
      2. spline_net: Spline-producing layers (residual block linear layers, resize layers)
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
        is_context = ("context_layer" in name) or ("initial_layer" in name)
        if is_context:
            param_groups["context_proj"][name] = param
        elif "transform_net" in name or "blocks" in name or "resize_layers" in name:
            param_groups["spline_net"][name] = param
        else:
            param_groups["remaining_flow"][name] = param
    return param_groups

def compute_relative_updates(initial_params, current_params, epsilon=1e-8):
    if not initial_params:
        return 0.0
    total_delta_sq = 0.0
    total_initial_sq = 0.0
    for name, p_init in initial_params.items():
        p_curr = current_params[name]
        delta = p_curr - p_init
        total_delta_sq += (delta.norm() ** 2).item()
        total_initial_sq += (p_init.norm() ** 2).item()
    return np.sqrt(total_delta_sq) / (np.sqrt(total_initial_sq) + epsilon)

def compute_grad_norm(params):
    if not params:
        return 0.0
    total_grad_sq = 0.0
    for name, p in params.items():
        if p.grad is not None:
            total_grad_sq += (p.grad.norm() ** 2).item()
    return np.sqrt(total_grad_sq)

def load_data_and_embedding(batch_size, device):
    """
    Loads real training data with beta0_true and chirp_mass, and computes 128D waveform embeddings.
    Strictly omits OnlineBeyondGRRotation and SampleBeyondGRProxy.
    """
    model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
    settings_path = "/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml"
    
    with open(settings_path, "r") as f:
        train_settings = yaml.safe_load(f)
        
    data_settings = copy.deepcopy(train_settings["data"])
    data_settings["inference_parameters"] = ["beta0_true", "chirp_mass"]
    data_settings["context_parameters"] = ["mass_ratio", "chi1z", "chi2z", "luminosity_distance", "theta_jn"]
    
    wfd_path = data_settings["waveform_dataset_path"]
    asd_path = train_settings["training"]["stage_0"]["asd_dataset_path"]
    
    print(f"\n[Data] Loading WaveformDataset from: {wfd_path}")
    wfd = WaveformDataset(file_name=wfd_path)
    
    # Strictly omit rotation and proxy transforms
    set_train_transforms(
        wfd=wfd,
        data_settings=data_settings,
        asd_dataset_path=asd_path,
        omit_transforms=[OnlineBeyondGRRotation, SampleBeyondGRProxy],
        print_output=False,
    )
    
    loader = DataLoader(wfd, batch_size=batch_size, shuffle=False, num_workers=0)
    raw_batch = next(iter(loader))
    
    # Extract tensors
    theta_raw = raw_batch[0].to(device).float()       # [B, 2]: [beta0_true, chirp_mass]
    wf_batch = raw_batch[1].to(device).float()        # [B, 207, 48]
    ctx_5params = raw_batch[2].to(device).float()     # [B, 5]: [q, chi1z, chi2z, dL, theta_jn]
    pos_batch = raw_batch[3].to(device).float()       # [B, 207, 3]
    mask_batch = raw_batch[4].to(device).bool()       # [B, 207]
    
    # Load pretrained embedding model
    print(f"[Embedding] Loading pretrained embedding network from: {model_path}")
    embedding_model = build_model_from_kwargs(
        filename=model_path,
        pretraining=False,
        pretrained_embedding_net=None,
        load_embedding_only=True,
        device=device,
        print_output=False,
    )
    embedding_net = embedding_model.network.embedding_net
    embedding_net.freeze_all_except_last_n_layers(n=0)
    embedding_net.eval()
    for p in embedding_net.parameters():
        p.requires_grad = False
        
    with torch.no_grad():
        emb_batch = embedding_net(wf_batch, pos_batch, mask_batch)
        if isinstance(emb_batch, tuple):
            emb_batch = emb_batch[0]
            
    B = theta_raw.shape[0]
    beta0_unstd = wfd.parameters.iloc[:B]["beta0_true"].values
    
    # Verification checks
    print(f"\n[Verification]")
    print(f"  Batch size               : {B}")
    print(f"  Raw beta0_true range     : [{beta0_unstd.min():.4f}, {beta0_unstd.max():.4f}] (Expected ~ [-5, +5])")
    print(f"  Standardized theta shape : {theta_raw.shape} -> col0: beta0, col1: chirp_mass")
    print(f"  Waveform embedding shape : {emb_batch.shape} (128-D)")
    print(f"  Context 5-params shape   : {ctx_5params.shape} (5-D: q, chi1z, chi2z, dL, theta_jn)")
    
    assert "beta_proxy" not in data_settings["context_parameters"], "CRITICAL: beta_proxy found in context parameters!"
    assert "beta_residual" not in data_settings["inference_parameters"], "CRITICAL: beta_residual found in inference parameters!"
    assert ctx_5params.shape[-1] == 5, f"Expected 5 context parameters, got {ctx_5params.shape[-1]}"
    print("  Explicit check: NO beta_proxy or beta_residual present in dataset or conditioning path.")
    
    return {
        "theta_2d": theta_raw,
        "theta_1d": theta_raw[:, :1],
        "emb": emb_batch,
        "ctx_5params": ctx_5params,
        "wfd": wfd,
        "B": B,
        "beta0_raw": torch.tensor(beta0_unstd, device=device, dtype=torch.float32),
    }

def run_test(test_id, data, args, device):
    """
    Executes one of the three controlled tests:
      Test 1: q(beta0 | embedding)                       [1D target, 128D context]
      Test 2: q(beta0, chirp_mass | embedding)            [2D target, 128D context]
      Test 3: q(beta0, chirp_mass | embedding, 5 params) [2D target, 133D context]
    """
    B = data["B"]
    checkpoints = sorted(list(set([0, 1, 25, 50, 100, 200, 300, args.num_steps])))
    
    # 1. Configure test specifications
    if test_id == "1":
        test_title = "TEST 1: q(beta0 | embedding)"
        target_dim = 1
        context_dim = 128
        theta = data["theta_1d"]
        context = data["emb"]
        target_names = ["beta0"]
    elif test_id == "2":
        test_title = "TEST 2: q(beta0, chirp_mass | embedding)"
        target_dim = 2
        context_dim = 128
        theta = data["theta_2d"]
        context = data["emb"]
        target_names = ["beta0", "chirp_mass"]
    elif test_id == "3":
        test_title = "TEST 3: q(beta0, chirp_mass | embedding, 5 context parameters)"
        target_dim = 2
        context_dim = 133  # 128 + 5
        theta = data["theta_2d"]
        context = torch.cat([data["emb"], data["ctx_5params"]], dim=-1)
        target_names = ["beta0", "chirp_mass"]
    else:
        raise ValueError(f"Unknown test_id {test_id}")
        
    print("\n" + "=" * 90)
    print(f"{test_title}")
    print("=" * 90)
    print(f"Target dimension : {target_dim} ({', '.join(target_names)})")
    print(f"Context dimension: {context_dim}")
    print(f"Flow steps       : {args.num_flow_steps}")
    print(f"Hidden dimension : {args.hidden_dim}")
    print(f"Transform blocks : {args.num_transform_blocks}")
    print(f"Spline bins      : {args.num_bins}")
    print(f"Batch size       : {B}")
    print(f"Optimizer steps  : {args.num_steps}")
    
    # 2. Build Freshly Initialized NSF Flow
    torch.manual_seed(args.seed)
    base_kwargs = {
        "hidden_dim": args.hidden_dim,
        "num_transform_blocks": args.num_transform_blocks,
        "activation": "elu",
        "batch_norm": False,
        "layer_norm": True,
        "dropout_probability": 0.0,
        "num_bins": args.num_bins,
        "base_transform_type": "rq-coupling",
        "context_in_initial_layer": True,
    }
    
    flow = create_nsf_model(
        input_dim=target_dim,
        context_dim=context_dim,
        num_flow_steps=args.num_flow_steps,
        base_transform_kwargs=base_kwargs,
    ).to(device)
    
    # 3. Parameter Categorization & Initial Snapshot
    param_groups = categorize_parameters(flow)
    initial_weights = {name: p.clone().detach() for name, p in flow.named_parameters() if p.requires_grad}
    initial_grouped = {
        g: {name: p.clone().detach() for name, p in group.items()}
        for g, group in param_groups.items()
    }
    
    optimizer = torch.optim.AdamW(
        flow.parameters(),
        lr=args.lr,
        betas=(0.8, 0.99),
        weight_decay=args.weight_decay,
    )
    
    # Fixed derangements for context shuffling
    perm_full = generate_derangement(B, device)
    perm_emb = generate_derangement(B, device)
    perm_params = generate_derangement(B, device)
    
    history = []
    
    def evaluate(step):
        flow.eval()
        with torch.no_grad():
            # A. Matched NLL
            lp_matched = flow.log_prob(theta, context)
            nll_matched = -lp_matched.mean().item()
            
            # B. Full Shuffled NLL
            lp_shuffled = flow.log_prob(theta, context[perm_full])
            nll_shuffled = -lp_shuffled.mean().item()
            delta = nll_shuffled - nll_matched
            
            # C. Decomposed shuffles for Test 3
            nll_shuf_emb = None
            nll_shuf_params = None
            delta_shuf_emb = None
            delta_shuf_params = None
            if test_id == "3":
                # Shuffle embedding only, keep 5 parameters matched
                ctx_shuf_emb = torch.cat([data["emb"][perm_emb], data["ctx_5params"]], dim=-1)
                lp_se = flow.log_prob(theta, ctx_shuf_emb)
                nll_shuf_emb = -lp_se.mean().item()
                delta_shuf_emb = nll_shuf_emb - nll_matched
                
                # Shuffle 5 parameters only, keep embedding matched
                ctx_shuf_params = torch.cat([data["emb"], data["ctx_5params"][perm_params]], dim=-1)
                lp_sp = flow.log_prob(theta, ctx_shuf_params)
                nll_shuf_params = -lp_sp.mean().item()
                delta_shuf_params = nll_shuf_params - nll_matched
                
        # Parameter updates
        current_grouped = {
            g: {name: p.detach() for name, p in group.items()}
            for g, group in param_groups.items()
        }
        upd_ctx = compute_relative_updates(initial_grouped["context_proj"], current_grouped["context_proj"])
        upd_spline = compute_relative_updates(initial_grouped["spline_net"], current_grouped["spline_net"])
        upd_flow = compute_relative_updates(initial_grouped["remaining_flow"], current_grouped["remaining_flow"])
        
        return {
            "step": step,
            "matched_nll": nll_matched,
            "shuffled_nll": nll_shuffled,
            "delta": delta,
            "nll_shuf_emb": nll_shuf_emb,
            "delta_shuf_emb": delta_shuf_emb,
            "nll_shuf_params": nll_shuf_params,
            "delta_shuf_params": delta_shuf_params,
            "upd_ctx": upd_ctx,
            "upd_spline": upd_spline,
            "upd_flow": upd_flow,
        }
        
    # Initial evaluation
    print(f"\n{'Step':>6} | {'Matched NLL':>12} | {'Shuffled NLL':>13} | {'Delta (Shuf-Mat)':>17} | "
          f"{'|Grad(Ctx)|':>12} | {'Upd(Ctx)':>10} | {'Upd(Spline)':>11} | {'Upd(Flow)':>10}")
    print("-" * 105)
    
    init_res = evaluate(0)
    init_res["grad_ctx"] = 0.0
    init_res["grad_spline"] = 0.0
    init_res["grad_flow"] = 0.0
    history.append(init_res)
    print(f"{0:>6d} | {init_res['matched_nll']:>12.6f} | {init_res['shuffled_nll']:>13.6f} | "
          f"{init_res['delta']:>+17.6f} | {'---':>12} | {init_res['upd_ctx']:>10.6f} | "
          f"{init_res['upd_spline']:>11.6f} | {init_res['upd_flow']:>10.6f}")
          
    # 4. Training Loop
    for step in range(1, args.num_steps + 1):
        flow.train()
        optimizer.zero_grad()
        
        lp = flow.log_prob(theta, context)
        loss = -lp.mean()
        loss.backward()
        
        grad_ctx = compute_grad_norm(param_groups["context_proj"])
        grad_spline = compute_grad_norm(param_groups["spline_net"])
        grad_flow = compute_grad_norm(param_groups["remaining_flow"])
        
        optimizer.step()
        
        if step in checkpoints:
            res = evaluate(step)
            res["grad_ctx"] = grad_ctx
            res["grad_spline"] = grad_spline
            res["grad_flow"] = grad_flow
            history.append(res)
            print(f"{step:>6d} | {res['matched_nll']:>12.6f} | {res['shuffled_nll']:>13.6f} | "
                  f"{res['delta']:>+17.6f} | {grad_ctx:>12.6e} | {res['upd_ctx']:>10.6f} | "
                  f"{res['upd_spline']:>11.6f} | {res['upd_flow']:>10.6f}")
                  
    final_res = history[-1]
    
    # 5. Target Localization & Output Statistics Audit
    # Evaluate sample predictions and conditional dependence localization
    flow.eval()
    with torch.no_grad():
        # Draw samples from flow conditioned on matched context
        # flow.sample(num_samples, context) -> [num_samples, B, target_dim] or [B, target_dim]
        # In nflows: flow.sample(1, context=context) -> [B, 1, target_dim]
        samples_matched = flow.sample(1, context=context).squeeze(1) # [B, target_dim]
        samples_shuffled = flow.sample(1, context=context[perm_full]).squeeze(1) # [B, target_dim]
        
    print("\n" + "-" * 90)
    print(f"TARGET LOCALIZATION & SENSITIVITY AUDIT ({test_title})")
    print("-" * 90)
    
    target_localization = {}
    for t_idx, t_name in enumerate(target_names):
        true_vals = theta[:, t_idx].cpu().numpy()
        pred_matched = samples_matched[:, t_idx].cpu().numpy()
        pred_shuffled = samples_shuffled[:, t_idx].cpu().numpy()
        
        corr_matched = np.corrcoef(true_vals, pred_matched)[0, 1]
        corr_shuffled = np.corrcoef(true_vals, pred_shuffled)[0, 1]
        mean_abs_err_matched = np.mean(np.abs(true_vals - pred_matched))
        mean_abs_err_shuffled = np.mean(np.abs(true_vals - pred_shuffled))
        
        # Output variation across different contexts for the SAME latent code
        # Feed identical latent z into inverse transform under different contexts
        z_fixed = torch.zeros(B, target_dim, device=device)
        y_c1, _ = flow._transform.inverse(z_fixed, context=context)
        y_c2, _ = flow._transform.inverse(z_fixed, context=context[perm_full])
        ctx_response = (y_c1[:, t_idx] - y_c2[:, t_idx]).abs().mean().item()
        
        target_localization[t_name] = {
            "corr_matched": corr_matched,
            "corr_shuffled": corr_shuffled,
            "mae_matched": mean_abs_err_matched,
            "mae_shuffled": mean_abs_err_shuffled,
            "ctx_response": ctx_response,
        }
        
        print(f"Target: {t_name}")
        print(f"  Matched correlation r(true, pred)  : {corr_matched:+.4f} (MAE: {mean_abs_err_matched:.4f})")
        print(f"  Shuffled correlation r(true, pred) : {corr_shuffled:+.4f} (MAE: {mean_abs_err_shuffled:.4f})")
        print(f"  Mean |Delta output| under context change: {ctx_response:.6f}")
        
    if test_id == "3":
        print("\nTest 3 Context Conditioning Decomposed Ablation:")
        print(f"  Delta when shuffling FULL context (Emb + 5 params) : {final_res['delta']:>+10.4f}")
        print(f"  Delta when shuffling EMBEDDING ONLY (params matched) : {final_res['delta_shuf_emb']:>+10.4f}")
        print(f"  Delta when shuffling 5 PARAMS ONLY (emb matched)     : {final_res['delta_shuf_params']:>+10.4f}")
        
        # Check specific influence of each component on beta0
        z_fixed = torch.zeros(B, target_dim, device=device)
        y_base, _ = flow._transform.inverse(z_fixed, context=context)
        ctx_only_emb = torch.cat([data["emb"][perm_emb], data["ctx_5params"]], dim=-1)
        ctx_only_params = torch.cat([data["emb"], data["ctx_5params"][perm_params]], dim=-1)
        y_emb_mod, _ = flow._transform.inverse(z_fixed, context=ctx_only_emb)
        y_param_mod, _ = flow._transform.inverse(z_fixed, context=ctx_only_params)
        
        beta0_resp_emb = (y_emb_mod[:, 0] - y_base[:, 0]).abs().mean().item()
        beta0_resp_params = (y_param_mod[:, 0] - y_base[:, 0]).abs().mean().item()
        print(f"  beta0 mean |Delta output| from Waveform Embedding change: {beta0_resp_emb:.6f}")
        print(f"  beta0 mean |Delta output| from 5 Context Params change  : {beta0_resp_params:.6f}")
        
    # 6. Verdict Determination
    # Criteria:
    # - "learned": Delta > 0.5 and matched correlation > 0.3
    # - "weakly learned": Delta between 0.05 and 0.5, or positive Delta with weak correlation
    # - "not learned": Delta < 0.05
    final_delta = final_res["delta"]
    if final_delta >= 0.5:
        verdict = "LEARNED"
    elif final_delta >= 0.05:
        verdict = "WEAKLY LEARNED"
    else:
        verdict = "NOT LEARNED"
        
    print(f"\n>>> VERDICT FOR {test_title}: {verdict} (Delta = {final_delta:+.4f})")
    
    return {
        "test_id": test_id,
        "title": test_title,
        "init_matched_nll": history[0]["matched_nll"],
        "init_shuffled_nll": history[0]["shuffled_nll"],
        "init_delta": history[0]["delta"],
        "final_matched_nll": final_res["matched_nll"],
        "final_shuffled_nll": final_res["shuffled_nll"],
        "final_delta": final_delta,
        "grad_ctx": final_res["grad_ctx"],
        "upd_ctx": final_res["upd_ctx"],
        "upd_spline": final_res["upd_spline"],
        "upd_flow": final_res["upd_flow"],
        "target_localization": target_localization,
        "verdict": verdict,
        "history": history,
        "final_res": final_res,
    }

def main():
    args = parse_args()
    if args.device is not None:
        device = args.device
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = "xpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")
    
    data = load_data_and_embedding(args.batch_size, device)
    
    tests_to_run = ["1", "2", "3"] if args.test == "all" else [args.test]
    results = {}
    
    for t_id in tests_to_run:
        results[t_id] = run_test(t_id, data, args, device)
        
    # Print Comprehensive Final Comparative Report
    print("\n" + "=" * 95)
    print("CONTROLLED DIAGNOSTIC SUITE: FINAL COMPARATIVE REPORT")
    print("=" * 95)
    print(f"{'Metric':<38} | {'Test 1: q(beta0|emb)':<16} | {'Test 2: q(b0,cm|emb)':<16} | {'Test 3: q(b0,cm|emb,5p)':<16}")
    print("-" * 95)
    
    def get_val(tid, key, fmt="{:.4f}"):
        if tid in results:
            v = results[tid][key]
            return fmt.format(v) if isinstance(v, (int, float)) else str(v)
        return "N/A"
        
    print(f"{'Target Dimension':<38} | {get_val('1', 'title', lambda x: '1D'):<16} | {get_val('2', 'title', lambda x: '2D'):<16} | {get_val('3', 'title', lambda x: '2D'):<16}")
    print(f"{'Context Dimension':<38} | {'128D':<16} | {'128D':<16} | {'133D':<16}")
    print(f"{'Initial Matched NLL':<38} | {get_val('1', 'init_matched_nll'):<16} | {get_val('2', 'init_matched_nll'):<16} | {get_val('3', 'init_matched_nll'):<16}")
    print(f"{'Final Matched NLL':<38} | {get_val('1', 'final_matched_nll'):<16} | {get_val('2', 'final_matched_nll'):<16} | {get_val('3', 'final_matched_nll'):<16}")
    print(f"{'Final Shuffled NLL':<38} | {get_val('1', 'final_shuffled_nll'):<16} | {get_val('2', 'final_shuffled_nll'):<16} | {get_val('3', 'final_shuffled_nll'):<16}")
    print(f"{'Final Delta (Shuffled - Matched)':<38} | {get_val('1', 'final_delta', '{:+.4f}'):<16} | {get_val('2', 'final_delta', '{:+.4f}'):<16} | {get_val('3', 'final_delta', '{:+.4f}'):<16}")
    print(f"{'Conditioner Grad Norm':<38} | {get_val('1', 'grad_ctx', '{:.2e}'):<16} | {get_val('2', 'grad_ctx', '{:.2e}'):<16} | {get_val('3', 'grad_ctx', '{:.2e}'):<16}")
    print(f"{'Conditioner Relative Update':<38} | {get_val('1', 'upd_ctx'):<16} | {get_val('2', 'upd_ctx'):<16} | {get_val('3', 'upd_ctx'):<16}")
    print(f"{'Spline Net Relative Update':<38} | {get_val('1', 'upd_spline'):<16} | {get_val('2', 'upd_spline'):<16} | {get_val('3', 'upd_spline'):<16}")
    print(f"{'Flow (Non-Ctx) Relative Update':<38} | {get_val('1', 'upd_flow'):<16} | {get_val('2', 'upd_flow'):<16} | {get_val('3', 'upd_flow'):<16}")
    
    # Target localization comparisons
    print("-" * 95)
    print("Target Localization:")
    for tid in tests_to_run:
        r = results[tid]
        print(f"  [{r['title']}]")
        for t_name, loc in r["target_localization"].items():
            print(f"    - {t_name:<11}: r(true, pred)={loc['corr_matched']:+.4f} (shuffled={loc['corr_shuffled']:+.4f}) | ctx_response={loc['ctx_response']:.4f}")
            
    print("-" * 95)
    print(f"{'FINAL VERDICT':<38} | {get_val('1', 'verdict'):<16} | {get_val('2', 'verdict'):<16} | {get_val('3', 'verdict'):<16}")
    print("=" * 95)

if __name__ == "__main__":
    main()
