"""
Diagnostic script to investigate why different beta_proxy chains produce nearly identical posterior distributions.
Covers:
  1. Transformer survival (waveform & embedding statistics, pairwise diffs, cosine similarity)
  2. Stage-by-stage preprocessing tracking (at what stage does proxy signal change/reduce)
  3. NSF context & log_prob sensitivity (fixed samples test, context delta norm)
  4. Control test & final summary
"""

import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/deathstar/dingorep/dingo")

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.inference.gw_samplers import BeyondGRSampler, create_sampler
from dingo.gw.transforms.beyond_gr_transforms import OnlineBeyondGRRotation

# -------------------------------------------------------------------------
# Load Model & Event Data
# -------------------------------------------------------------------------
model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/model_latest.pt"
event_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/GW190701_203306_event_data.hdf5"

print("=" * 80)
print(f"LOADING MODEL: {model_path}")
print("=" * 80)

model = build_model_from_kwargs(filename=model_path, device="cpu", load_training_info=False)
event_dataset = EventDataset(file_name=event_path)

sampler = create_sampler(model=model)
sampler.context = event_dataset.data
sampler.event_metadata = event_dataset.settings

proxies = [-4.0, -2.0, 0.0, 2.0, 4.0]
base_context = copy.deepcopy(sampler.context)
base_parameters = sampler._build_chain_parameters()

print(f"Detectors: {sampler.detectors}")
print(f"Dingo-T1 chirp_mass median: {sampler.dingo_medians['chirp_mass']:.4f} M_sun")
print(f"Proxies: {proxies}")

# =========================================================================
# INVESTIGATION 1: DOES THE PROXY INFORMATION SURVIVE THE TRANSFORMER?
# =========================================================================
print("\n" + "=" * 80)
print("INVESTIGATION 1: DOES PROXY INFORMATION SURVIVE THE TRANSFORMER?")
print("=" * 80)

waveforms_dict = {}
embeddings_dict = {}
context_vectors_dict = {}

model.network.eval()

for proxy in proxies:
    ctx = copy.deepcopy(base_context)
    ctx["parameters"] = copy.deepcopy(base_parameters)
    ctx.setdefault("extrinsic_parameters", {})
    ctx["extrinsic_parameters"]["beta_proxy"] = float(proxy)

    x = sampler.transform_pre(ctx)
    if isinstance(x, (list, tuple)):
        x_batched = [xi.unsqueeze(0) for xi in x]
    else:
        x_batched = [x.unsqueeze(0)]

    # x_batched[0] is waveform [1, num_tokens, features]
    wf = x_batched[0]
    with torch.no_grad():
        ctx_vec, _ = model.network._get_context(*x_batched)

    embed = ctx_vec[:, :128].float()
    
    waveforms_dict[proxy] = wf.detach().float().squeeze(0) # [num_tokens, features]
    embeddings_dict[proxy] = embed.detach().float().squeeze(0) # [128]
    context_vectors_dict[proxy] = ctx_vec.detach().float().squeeze(0) # [134]

    wf_norm = torch.linalg.vector_norm(waveforms_dict[proxy]).item()
    emb_norm = torch.linalg.vector_norm(embeddings_dict[proxy]).item()

    print(f"\n--- Proxy {proxy:+.1f} ---")
    print(f"  Waveform  shape={tuple(waveforms_dict[proxy].shape)}, mean={waveforms_dict[proxy].mean().item():.6e}, std={waveforms_dict[proxy].std().item():.6e}, min={waveforms_dict[proxy].min().item():.6e}, max={waveforms_dict[proxy].max().item():.6e}, norm={wf_norm:.6e}")
    print(f"  Embedding shape={tuple(embeddings_dict[proxy].shape)}, mean={embeddings_dict[proxy].mean().item():.6e}, std={embeddings_dict[proxy].std().item():.6e}, min={embeddings_dict[proxy].min().item():.6e}, max={embeddings_dict[proxy].max().item():.6e}, norm={emb_norm:.6e}")

print("\n--- Pairwise Waveform and Embedding Differences (All Pairs) ---")
print(f"{'Pair (p_i, p_j)':<18} | {'||W_i - W_j||':<14} | {'Rel Diff (W)':<14} | {'||E_i - E_j||':<14} | {'Rel Diff (E)':<14} | {'Cosine Sim (E)':<14}")
print("-" * 96)

for i in range(len(proxies)):
    for j in range(i + 1, len(proxies)):
        pi, pj = proxies[i], proxies[j]
        wi, wj = waveforms_dict[pi], waveforms_dict[pj]
        ei, ej = embeddings_dict[pi], embeddings_dict[pj]

        w_diff = torch.linalg.vector_norm(wi - wj).item()
        w_rel = w_diff / (torch.linalg.vector_norm(wi).item() + 1e-12)

        e_diff = torch.linalg.vector_norm(ei - ej).item()
        e_rel = e_diff / (torch.linalg.vector_norm(ei).item() + 1e-12)

        cos_sim = F.cosine_similarity(ei.unsqueeze(0), ej.unsqueeze(0)).item()

        print(f"({pi:+.1f}, {pj:+.1f}){'':<6} | {w_diff:<14.6e} | {w_rel:<14.6e} | {e_diff:<14.6e} | {e_rel:<14.6e} | {cos_sim:<14.8f}")

# =========================================================================
# INVESTIGATION 2: DOES PREPROCESSING/TOKENIZATION REMOVE THE PROXY SIGNAL?
# =========================================================================
print("\n" + "=" * 80)
print("INVESTIGATION 2: STAGE-BY-STAGE PREPROCESSING TRACKING")
print("=" * 80)

transforms_list = sampler.transform_pre.transforms
print(f"Total transforms in pipeline: {len(transforms_list)}")
for idx, t in enumerate(transforms_list):
    print(f"  Stage {idx}: {type(t).__name__}")

def extract_stage_tensor(sample, stage_name):
    """Extract a representative numpy array or tensor from the sample for comparison."""
    if isinstance(sample, (list, tuple)):
        # Unpacked tuple
        return sample[0].detach().cpu().numpy() if isinstance(sample[0], torch.Tensor) else sample[0]
    if isinstance(sample, dict):
        if "waveform" in sample:
            wf = sample["waveform"]
            if isinstance(wf, dict):
                # Dict of detector strains {H1: ..., L1: ..., V1: ...}
                # Concatenate all detectors for comparison
                return np.concatenate([np.asarray(wf[k]) for k in sorted(wf.keys())])
            elif isinstance(wf, (np.ndarray, torch.Tensor)):
                return wf.detach().cpu().numpy() if isinstance(wf, torch.Tensor) else wf
        if "strains" in sample:
            return sample["strains"]
    return None

stage_results = {idx: {} for idx in range(len(transforms_list))}

for proxy in proxies:
    current_sample = copy.deepcopy(base_context)
    current_sample["parameters"] = copy.deepcopy(base_parameters)
    current_sample.setdefault("extrinsic_parameters", {})
    current_sample["extrinsic_parameters"]["beta_proxy"] = float(proxy)

    for idx, t in enumerate(transforms_list):
        current_sample = t(current_sample)
        arr = extract_stage_tensor(current_sample, type(t).__name__)
        stage_results[idx][proxy] = arr

print("\n--- Stage-by-Stage Statistics & Pairwise Relative Differences ---")

for idx, t in enumerate(transforms_list):
    t_name = type(t).__name__
    print(f"\n=======================================================")
    print(f"STAGE {idx}: {t_name}")
    print(f"=======================================================")

    for proxy in proxies:
        arr = stage_results[idx][proxy]
        if arr is not None:
            norm_val = np.linalg.norm(arr)
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            max_abs = np.max(np.abs(arr))
            print(f"  Proxy {proxy:+.1f}: shape={arr.shape}, dtype={arr.dtype}, norm={norm_val:.6e}, mean={mean_val:.6e}, std={std_val:.6e}, max_abs={max_abs:.6e}")
        else:
            print(f"  Proxy {proxy:+.1f}: Non-tensor/dict output")

    # Pairwise comparison relative to proxy -4.0 and extreme pair (-4, +4)
    arr_ref = stage_results[idx][-4.0]
    if arr_ref is not None:
        norm_ref = np.linalg.norm(arr_ref)
        print(f"\n  Pairwise Differences relative to Proxy -4.0:")
        for proxy in proxies[1:]:
            arr_p = stage_results[idx][proxy]
            diff = np.linalg.norm(arr_p - arr_ref)
            rel_diff = diff / (norm_ref + 1e-12)
            print(f"    (-4.0 vs {proxy:+.1f}): ||Δ|| = {diff:.6e} | Rel Δ = {rel_diff:.6e}")

# =========================================================================
# INVESTIGATION 3 & 4: DOES THE NSF ACTUALLY USE THE 134-D CONTEXT?
# =========================================================================
print("\n" + "=" * 80)
print("INVESTIGATION 3 & 4: NSF CONTEXT & LOG_PROB SENSITIVITY TEST")
print("=" * 80)

# Check context vectors
print("\n--- Context Vector Verification ---")
for proxy in proxies:
    cvec = context_vectors_dict[proxy] # [134]
    emb = cvec[:128]
    c_params = cvec[128:134]
    print(f"Proxy {proxy:+.1f}: context.shape={tuple(cvec.shape)}")
    print(f"  beta_proxy column (standardized): {c_params[0].item():.6f}")
    print(f"  fixed context parameters        : {c_params[1:].numpy()}")
    print(f"  embedding norm={torch.linalg.vector_norm(emb).item():.6f}, full context norm={torch.linalg.vector_norm(cvec).item():.6f}")

# Context Sensitivity Delta (C_-4 vs C_+4)
c_neg4 = context_vectors_dict[-4.0]
c_pos4 = context_vectors_dict[4.0]
delta_c = c_pos4 - c_neg4

print("\n--- Direct Context Sensitivity (Proxy +4.0 vs -4.0) ---")
print(f"||C_+4 - C_-4|| (total 134-D)    : {torch.linalg.vector_norm(delta_c).item():.6e}")
print(f"||delta[:, :128]|| (embedding)   : {torch.linalg.vector_norm(delta_c[:128]).item():.6e}")
print(f"||delta[:, 128:]|| (params)      : {torch.linalg.vector_norm(delta_c[128:]).item():.6e}")
print(f"delta in 6 context parameters    : {delta_c[128:].numpy()}")

# Controlled Sensitivity Test with FIXED samples S
print("\n--- Controlled Sensitivity Test: flow.log_prob(S, context_i) with FIXED S ---")

# Generate 1000 fixed standard normal samples (in flow target space: 2D)
torch.manual_seed(42)
num_test_samples = 1000
# Also let's draw samples from the base flow distribution or sample from proxy 0
S_fixed = torch.randn(num_test_samples, 2)

log_probs_dict = {}

flow_net = model.network.flow # The 2D NSF flow

for proxy in proxies:
    cvec = context_vectors_dict[proxy].unsqueeze(0) # [1, 134]
    # Expand context for num_test_samples
    cvec_expanded = cvec.expand(num_test_samples, -1)

    with torch.no_grad():
        lp = flow_net.log_prob(S_fixed, cvec_expanded) # [num_test_samples]
    
    log_probs_dict[proxy] = lp
    print(f"Proxy {proxy:+.1f}: log_prob mean={lp.mean().item():.6f}, std={lp.std().item():.6f}, min={lp.min().item():.6f}, max={lp.max().item():.6f}")

print("\n--- Pairwise log_prob Differences across Proxies (Fixed S) ---")
print(f"{'Pair (p_i, p_j)':<18} | {'Mean |lp_i - lp_j|':<20} | {'Max |lp_i - lp_j|':<20} | {'Correlation':<15}")
print("-" * 78)

for i in range(len(proxies)):
    for j in range(i + 1, len(proxies)):
        pi, pj = proxies[i], proxies[j]
        lpi, lpj = log_probs_dict[pi], log_probs_dict[pj]

        diff = torch.abs(lpi - lpj)
        mean_abs_diff = diff.mean().item()
        max_abs_diff = diff.max().item()

        # Correlation between log_probs
        corr = torch.corrcoef(torch.stack([lpi, lpj]))[0, 1].item()

        print(f"({pi:+.1f}, {pj:+.1f}){'':<6} | {mean_abs_diff:<20.6e} | {max_abs_diff:<20.6e} | {corr:<15.8f}")

# Also test log_prob with samples generated from proxy 0 flow
with torch.no_grad():
    c0 = context_vectors_dict[0.0].unsqueeze(0) # [1, 134]
    S_flow_samples = flow_net.sample(num_test_samples, c0) # shape might be [1, num_test_samples, 2]
    if S_flow_samples.ndim == 3:
        S_flow_samples = S_flow_samples.squeeze(0) # [num_test_samples, 2]

print(f"\n--- Sensitivity Test with S sampled from flow at Proxy 0.0 (S shape: {tuple(S_flow_samples.shape)}) ---")
log_probs_flow_samples = {}
for proxy in proxies:
    cvec_expanded = context_vectors_dict[proxy].unsqueeze(0).expand(num_test_samples, -1)
    with torch.no_grad():
        lp = flow_net.log_prob(S_flow_samples, cvec_expanded)
    log_probs_flow_samples[proxy] = lp
    print(f"Proxy {proxy:+.1f}: log_prob mean={lp.mean().item():.6f}, std={lp.std().item():.6f}")

print("\nPairwise log_prob differences on realistic flow samples:")
for i in range(len(proxies)):
    for j in range(i + 1, len(proxies)):
        pi, pj = proxies[i], proxies[j]
        lpi, lpj = log_probs_flow_samples[pi], log_probs_flow_samples[pj]
        diff = torch.abs(lpi - lpj)
        corr = torch.corrcoef(torch.stack([lpi, lpj]))[0, 1].item()
        print(f"  ({pi:+.1f}, {pj:+.1f}): Mean |Δlp| = {diff.mean().item():.6e}, Max |Δlp| = {diff.max().item():.6e}, Corr = {corr:.8f}")

# What if we isolate embedding vs context_parameters?
print("\n--- Ablation: Does NSF respond to embedding change vs context_parameters change? ---")
c_base = context_vectors_dict[0.0].unsqueeze(0).expand(num_test_samples, -1).clone()

# Change ONLY embedding to proxy +4, keep context_params at 0
c_only_emb = c_base.clone()
c_only_emb[:, :128] = context_vectors_dict[4.0][:128]

# Change ONLY context_parameters to proxy +4, keep embedding at 0
c_only_params = c_base.clone()
c_only_params[:, 128:] = context_vectors_dict[4.0][128:]

with torch.no_grad():
    lp_base = flow_net.log_prob(S_flow_samples, c_base)
    lp_only_emb = flow_net.log_prob(S_flow_samples, c_only_emb)
    lp_only_params = flow_net.log_prob(S_flow_samples, c_only_params)
    lp_both = flow_net.log_prob(S_flow_samples, context_vectors_dict[4.0].unsqueeze(0).expand(num_test_samples, -1))

print(f"Mean |Δlp| when changing ONLY Embedding (0 -> +4)     : {torch.abs(lp_only_emb - lp_base).mean().item():.6e}")
print(f"Mean |Δlp| when changing ONLY Context Params (0 -> +4): {torch.abs(lp_only_params - lp_base).mean().item():.6e}")
print(f"Mean |Δlp| when changing BOTH (0 -> +4)               : {torch.abs(lp_both - lp_base).mean().item():.6e}")

# Posterior sampling test: Draw 5000 samples for each proxy and compare medians/stds in physical space
print("\n--- Direct Posterior Sample Comparison across Proxies ---")
chains = sampler.generate_proxy_chains(num_samples=5000, batch_size=5000)

for proxy, smp in chains.items():
    beta_res = np.array(smp["beta_residual"])
    cm = np.array(smp["chirp_mass"])
    print(f"Proxy {proxy:+.1f}: beta_residual median={np.median(beta_res):.6f}, std={np.std(beta_res):.6f} | chirp_mass median={np.median(cm):.6f}, std={np.std(cm):.6f}")

print("\n" + "=" * 80)
print("DIAGNOSTIC RUN COMPLETE")
print("=" * 80)

