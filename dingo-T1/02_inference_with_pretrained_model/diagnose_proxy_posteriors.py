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
model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
event_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/GW190701_203306_event_data.hdf5"

print("=" * 80)
print(f"LOADING MODEL: {model_path}")
print("=" * 80)

model = build_model_from_kwargs(filename=model_path, device="xpu", load_training_info=False)
event_dataset = EventDataset(file_name=event_path)

# Always use BeyondGRSampler for the diagnostic pipeline
sampler = BeyondGRSampler(model=model)
sampler.context = event_dataset.data
sampler.event_metadata = event_dataset.settings

proxies = [-4.0, -2.0, 0.0, 2.0, 4.0]
base_context = copy.deepcopy(sampler.context)
base_parameters = sampler._build_chain_parameters()

print(f"Detectors: {sampler.detectors}")
print(f"Dingo-T1 chirp_mass median: {sampler.dingo_medians['chirp_mass']:.4f} M_sun")
print(f"Proxies: {proxies}")

# ============================================================================
# MYRESIDUALBLOCK CONTEXT ISOLATION TEST
# ============================================================================
#
# Purpose:
#   Determine whether changing the context actually changes the output of
#   MyResidualBlock, independently of the rest of the NSF.
#
# This automatically handles:
#   - embedding blocks: context dimension = 5
#   - flow blocks:      context dimension = 134
#
# ============================================================================

print("\n" + "=" * 80)
print("MYRESIDUALBLOCK CONTEXT ISOLATION TEST")
print("=" * 80)


def find_my_residual_blocks(module):
    """Return all MyResidualBlock modules with their full names."""
    blocks = []

    for name, submodule in module.named_modules():
        if submodule.__class__.__name__ == "MyResidualBlock":
            blocks.append((name, submodule))

    return blocks


# --------------------------------------------------------------------------
# Find every MyResidualBlock in the loaded model
# --------------------------------------------------------------------------

all_blocks = find_my_residual_blocks(model.network)

print(f"\nFound {len(all_blocks)} MyResidualBlock(s).")

for idx, (name, block) in enumerate(all_blocks):
    context_layer = getattr(block, "context_layer", None)

    if context_layer is not None:
        context_dim = context_layer.in_features
        output_dim = context_layer.out_features
    else:
        context_dim = None
        output_dim = None

    print(
        f"[{idx:02d}] {name} | "
        f"context_dim={context_dim} | "
        f"context_out={output_dim}"
    )


# ============================================================================
# Helper: run isolation test on ONE block
# ============================================================================

def isolate_my_residual_block(
    block,
    block_name,
    batch_size=8,
    sequence_length=207,
    device="xpu",
):
    """
    Test whether changing context changes MyResidualBlock output.

    The context dimensionality is inferred directly from block.context_layer.
    """

    print("\n" + "=" * 80)
    print(f"ISOLATION TEST: {block_name}")
    print("=" * 80)

    block.eval()

    # ----------------------------------------------------------------------
    # Determine architecture dimensions directly from the block
    # ----------------------------------------------------------------------

    context_layer = getattr(block, "context_layer", None)

    if context_layer is None:
        print("SKIPPED: block has no context_layer.")
        return None

    context_dim = context_layer.in_features

    # Determine feature dimension from the LayerNorm.
    # MyResidualBlock has LayerNorm(feature_dim).
    layer_norm = block.layer_norm_layers[0]
    feature_dim = layer_norm.normalized_shape[0]

    print(f"feature dimension : {feature_dim}")
    print(f"context dimension : {context_dim}")

    # ----------------------------------------------------------------------
    # Construct identical input X
    # ----------------------------------------------------------------------

    torch.manual_seed(12345)

    x = torch.randn(
        batch_size,
        sequence_length,
        feature_dim,
        device=device,
    )

    # ----------------------------------------------------------------------
    # Construct two DIFFERENT contexts
    #
    # IMPORTANT:
    # Same X is used for both.
    # Only context changes.
    # ----------------------------------------------------------------------

    context_zero = torch.zeros(
        batch_size,
        sequence_length,
        context_dim,
        device=device,
    )

    context_random = torch.randn(
        batch_size,
        sequence_length,
        context_dim,
        device=device,
    )

    # ----------------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------------

    with torch.no_grad():

        output_zero = block(
            x.clone(),
            context=context_zero,
        )

        output_random = block(
            x.clone(),
            context=context_random,
        )

    # ----------------------------------------------------------------------
    # Output sensitivity
    # ----------------------------------------------------------------------

    delta = output_random - output_zero

    mean_abs_delta = delta.abs().mean().item()
    max_abs_delta = delta.abs().max().item()
    delta_norm = torch.linalg.vector_norm(delta).item()

    output_zero_norm = torch.linalg.vector_norm(output_zero).item()
    output_random_norm = torch.linalg.vector_norm(output_random).item()

    relative_delta = (
        delta_norm /
        (torch.linalg.vector_norm(output_zero).item() + 1e-12)
    )

    print("\n--- Output sensitivity ---")
    print(f"output_zero shape       : {tuple(output_zero.shape)}")
    print(f"output_random shape     : {tuple(output_random.shape)}")
    print(f"mean |Δoutput|          : {mean_abs_delta:.8e}")
    print(f"max  |Δoutput|          : {max_abs_delta:.8e}")
    print(f"||Δoutput||             : {delta_norm:.8e}")
    print(f"relative ||Δoutput||    : {relative_delta:.8e}")

    # ----------------------------------------------------------------------
    # Direct context projection test
    #
    # This isolates the context_layer itself.
    # ----------------------------------------------------------------------

    with torch.no_grad():

        projected_zero = block.context_layer(
            context_zero
        )

        projected_random = block.context_layer(
            context_random
        )

    projected_delta = projected_random - projected_zero

    print("\n--- context_layer sensitivity ---")
    print(
        f"mean |Δ(context_layer)| : "
        f"{projected_delta.abs().mean().item():.8e}"
    )

    print(
        f"||Δ(context_layer)||    : "
        f"{torch.linalg.vector_norm(projected_delta).item():.8e}"
    )

    print(
        f"context_layer weight norm: "
        f"{torch.linalg.vector_norm(block.context_layer.weight).item():.8e}"
    )

    print(
        f"context_layer bias norm  : "
        f"{torch.linalg.vector_norm(block.context_layer.bias).item():.8e}"
    )

    # ----------------------------------------------------------------------
    # Gradient test
    #
    # This asks:
    # "Can loss/output mathematically transmit a gradient through context?"
    # ----------------------------------------------------------------------

    x_grad = x.clone().detach().requires_grad_(False)

    context_grad = context_random.clone().detach().requires_grad_(True)

    output_grad = block(
        x_grad,
        context=context_grad,
    )

    # Simple scalar objective
    diagnostic_loss = output_grad.pow(2).mean()

    diagnostic_loss.backward()

    grad = context_grad.grad

    print("\n--- Gradient through context ---")

    if grad is None:

        print("❌ NO GRADIENT REACHED CONTEXT")

        grad_mean = 0.0
        grad_norm = 0.0

    else:

        grad_mean = grad.abs().mean().item()
        grad_norm = torch.linalg.vector_norm(grad).item()

        print(
            f"mean |dL/dcontext|     : "
            f"{grad_mean:.8e}"
        )

        print(
            f"||dL/dcontext||        : "
            f"{grad_norm:.8e}"
        )

    # ----------------------------------------------------------------------
    # Final classification
    # ----------------------------------------------------------------------

    if mean_abs_delta < 1e-7:

        verdict = "❌ CONTEXT HAS NO EFFECT ON BLOCK OUTPUT"

    elif grad is None or grad_norm < 1e-10:

        verdict = "⚠️ OUTPUT RESPONDS BUT GRADIENT THROUGH CONTEXT IS EFFECTIVELY ZERO"

    else:

        verdict = "✅ BLOCK IS FUNCTIONALLY CONTEXT-CONDITIONAL"

    print("\n--- VERDICT ---")
    print(verdict)

    return {
        "name": block_name,
        "feature_dim": feature_dim,
        "context_dim": context_dim,
        "mean_abs_output_delta": mean_abs_delta,
        "max_abs_output_delta": max_abs_delta,
        "output_delta_norm": delta_norm,
        "relative_output_delta": relative_delta,
        "context_projection_delta_norm":
            torch.linalg.vector_norm(projected_delta).item(),
        "context_gradient_mean": grad_mean,
        "context_gradient_norm": grad_norm,
        "verdict": verdict,
    }


# ============================================================================
# Run the test
# ============================================================================

isolation_results = []

for block_name, block in all_blocks:

    result = isolate_my_residual_block(
        block=block,
        block_name=block_name,
        batch_size=8,
        sequence_length=207,
        device="cpu",
    )

    if result is not None:
        isolation_results.append(result)


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MYRESIDUALBLOCK ISOLATION SUMMARY")
print("=" * 80)

for result in isolation_results:

    print(
        f"\n{result['name']}"
    )

    print(
        f"  context_dim              = "
        f"{result['context_dim']}"
    )

    print(
        f"  mean |Δoutput|           = "
        f"{result['mean_abs_output_delta']:.6e}"
    )

    print(
        f"  relative ||Δoutput||     = "
        f"{result['relative_output_delta']:.6e}"
    )

    print(
        f"  ||Δcontext projection||  = "
        f"{result['context_projection_delta_norm']:.6e}"
    )

    print(
        f"  ||dL/dcontext||          = "
        f"{result['context_gradient_norm']:.6e}"
    )

    print(
        f"  VERDICT                  = "
        f"{result['verdict']}"
    )

print("\n" + "=" * 80)
print("ISOLATION TEST COMPLETE")
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
        if hasattr(model.network, "_get_context"):
            ctx_out = model.network._get_context(*x_batched)
            if isinstance(ctx_out, tuple):
                ctx_vec = ctx_out[0]
            else:
                ctx_vec = ctx_out
        elif hasattr(model.network, "embedding_net"):
            # Original FlowWrapper with TransformerEmbeddingNet
            wf_in = None
            c_params_in = None
            pos_in = None
            mask_in = None
            for item in x_batched:
                if isinstance(item, torch.Tensor):
                    if item.ndim == 3 and item.shape[-1] not in (3, 5):
                        wf_in = item
                    elif item.ndim == 3 and item.shape[-1] in (3, 5):
                        pos_in = item
                    elif item.ndim == 2 and item.shape[-1] == 6:
                        c_params_in = item
                    elif item.ndim == 2 and item.dtype == torch.bool:
                        mask_in = item

            if wf_in is None:
                wf_in = x_batched[0]
            if pos_in is None and len(x_batched) > 2:
                pos_in = x_batched[2]
            if mask_in is None and len(x_batched) > 3:
                mask_in = x_batched[3]
            if c_params_in is None:
                c_params_in = torch.tensor([[float(proxy), 0.7049, -0.135, -0.108, 2845.0, 1.58]])

            if pos_in is not None and mask_in is not None:
                emb_out = model.network.embedding_net(wf_in, pos_in, mask_in)
            elif pos_in is not None:
                emb_out = model.network.embedding_net(wf_in, pos_in)
            else:
                emb_out = model.network.embedding_net(wf_in)

            if isinstance(emb_out, tuple):
                emb_out = emb_out[0]
            ctx_vec = torch.cat([emb_out, c_params_in], dim=-1)
        else:
            raise AttributeError("model.network has neither _get_context nor embedding_net")

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

flow_net = model.network.flow # The NSF flow
flow_target_dim = flow_net._distribution._shape[0] if hasattr(flow_net, "_distribution") and hasattr(flow_net._distribution, "_shape") else 2

# Determine expected context dim for flow
flow_context_dim = 128
for m in flow_net.modules():
    if hasattr(m, "context_features") and m.context_features is not None:
        flow_context_dim = m.context_features
        break
    if hasattr(m, "context_layer") and hasattr(m.context_layer, "in_features"):
        flow_context_dim = m.context_layer.in_features
        break

print(f"Flow target dimension: {flow_target_dim}")
print(f"Flow context dimension: {flow_context_dim}")

print("\n" + "=" * 80)
print("TEST 1: DIRECT NSF TRANSFORMATION SENSITIVITY")
print("=" * 80)

torch.manual_seed(1234)

num_test = 2000

# Same physical-space target samples for every context.
y_fixed = torch.randn(num_test, flow_target_dim)

transform_outputs = {}
transform_logdets = {}

for proxy in proxies:

    if flow_context_dim == 128:
        ctx_data = embeddings_dict[proxy]
    else:
        ctx_data = context_vectors_dict[proxy]

    context = (
        ctx_data
        .unsqueeze(0)
        .expand(num_test, -1)
    )

    with torch.no_grad():
        z, logabsdet = flow_net._transform(
            y_fixed,
            context,
        )

    transform_outputs[proxy] = z.cpu()
    transform_logdets[proxy] = logabsdet.cpu()

    print(
        f"Proxy {proxy:+.1f}: "
        f"z_mean={z.mean().item():.8e}, "
        f"z_std={z.std().item():.8e}, "
        f"logdet_mean={logabsdet.mean().item():.8e}"
    )

reference_proxy = -4.0

print("\n--- Transformation differences relative to -4.0 ---")

for proxy in proxies[1:]:

    dz = (
        transform_outputs[proxy]
        - transform_outputs[reference_proxy]
    )

    dlogdet = (
        transform_logdets[proxy]
        - transform_logdets[reference_proxy]
    )

    print(
        f"-4.0 -> {proxy:+.1f}: "
        f"mean|Δz|={dz.abs().mean().item():.8e}, "
        f"max|Δz|={dz.abs().max().item():.8e}, "
        f"||Δz||={torch.linalg.vector_norm(dz).item():.8e}, "
        f"mean|Δlogdet|={dlogdet.abs().mean().item():.8e}"
    )

print("\n" + "=" * 80)
print("TEST 2: FIXED-LATENT POSTERIOR RESPONSE")
print("=" * 80)

torch.manual_seed(5678)

num_test = 5000

# Identical latent samples for every proxy.
z_fixed = torch.randn(num_test, flow_target_dim)

inverse_outputs = {}

for proxy in proxies:

    if flow_context_dim == 128:
        ctx_data = embeddings_dict[proxy]
    else:
        ctx_data = context_vectors_dict[proxy]

    context = (
        ctx_data
        .unsqueeze(0)
        .expand(num_test, -1)
    )

    with torch.no_grad():

        try:
            y, _ = flow_net._transform.inverse(
                z_fixed,
                context,
            )
        except (AttributeError, TypeError):
            # Fallback for implementations exposing inverse through transform
            y, _ = flow_net._transform.inverse(
                z_fixed,
                context=context,
            )

    inverse_outputs[proxy] = y.cpu()

    print(
        f"Proxy {proxy:+.1f}: "
        f"y_mean={y.mean().item():.8e}, "
        f"y_std={y.std().item():.8e}"
    )

print("\n--- Fixed-latent posterior differences ---")

ref = inverse_outputs[-4.0]

for proxy in proxies[1:]:

    delta = inverse_outputs[proxy] - ref

    print(
        f"-4.0 -> {proxy:+.1f}: "
        f"mean|ΔY|={delta.abs().mean().item():.8e}, "
        f"||ΔY||={torch.linalg.vector_norm(delta).item():.8e}"
    )

print("\n" + "=" * 80)
print("TEST 3: GRADIENT SENSITIVITY TO CONTEXT")
print("=" * 80)

torch.manual_seed(42)

y_probe = torch.randn(2000, flow_target_dim)

for proxy in proxies:

    if flow_context_dim == 128:
        ctx_data = embeddings_dict[proxy]
    else:
        ctx_data = context_vectors_dict[proxy]

    context = (
        ctx_data
        .clone()
        .detach()
        .unsqueeze(0)
        .expand(len(y_probe), -1)
        .clone()
        .requires_grad_(True)
    )

    log_prob = flow_net.log_prob(
        y_probe,
        context,
    )

    # Mean log probability gives a scalar.
    scalar = log_prob.mean()

    grad = torch.autograd.grad(
        scalar,
        context,
        retain_graph=False,
        create_graph=False,
    )[0]

    embedding_grad = grad[:, :min(128, grad.shape[-1])]

    print(
        f"\nProxy {proxy:+.1f}"
    )

    if grad.shape[-1] > 128:
        beta_grad = grad[:, 128]
        print(
            "  beta_proxy |grad| mean =",
            beta_grad.abs().mean().item()
        )
        print(
            "  beta_proxy |grad| max  =",
            beta_grad.abs().max().item()
        )

    print(
        "  embedding |grad| mean  =",
        embedding_grad.abs().mean().item()
    )

    print(
        "  full-context grad norm  =",
        torch.linalg.vector_norm(grad).item()
    )

print("\n" + "=" * 80)
print("\n" + "=" * 80)
print("TEST 4: CONTEXT SENSITIVITY ABLATION")
print("=" * 80)

contexts = {}

if flow_context_dim > 128:
    for proxy in proxies:
        context = context_vectors_dict[4.0].clone()
        context[128] = context_vectors_dict[proxy][128]
        contexts[proxy] = context
        print(
            f"Constructed context: "
            f"embedding from +4.0, "
            f"beta_proxy from {proxy:+.1f}"
        )
else:
    for proxy in proxies:
        contexts[proxy] = embeddings_dict[proxy]
        print(
            f"Constructed context: "
            f"128-D embedding from proxy {proxy:+.1f}"
        )

log_probs = {}

for proxy in proxies:

    context = (
        contexts[proxy]
        .unsqueeze(0)
        .expand(len(y_probe), -1)
    )

    with torch.no_grad():
        lp = flow_net.log_prob(
            y_probe,
            context,
        )

    log_probs[proxy] = lp

    print(
        f"proxy coordinate {proxy:+.1f}: "
        f"mean log_prob={lp.mean().item():.8e}"
    )

print("\n--- Context Effect relative to -4.0 ---")

for proxy in proxies[1:]:

    delta = (
        log_probs[proxy]
        - log_probs[-4.0]
    )

    print(
        f"-4.0 -> {proxy:+.1f}: "
        f"mean|Δlogp|={delta.abs().mean().item():.8e}, "
        f"max|Δlogp|={delta.abs().max().item():.8e}"
    )

print("\n" + "=" * 90)
print("TEST 5: INTERNAL SPLINE-CONDITIONER SENSITIVITY")
print("=" * 90)

torch.manual_seed(2026)

num_probe = 256
y_probe = torch.randn(num_probe, flow_target_dim)

coupling_modules = []

for name, module in flow_net.named_modules():

    if hasattr(module, "transform_net"):
        transform_net = getattr(module, "transform_net")

        if isinstance(transform_net, torch.nn.Module):
            coupling_modules.append(
                (name, module, transform_net)
            )

print(f"Found {len(coupling_modules)} modules with transform_net.")

captured = {}

def make_hook(layer_name):

    def hook(module, inputs, output):

        if isinstance(output, tuple):
            output = output[0]

        captured[layer_name] = (
            output.detach().float().cpu()
        )

    return hook

handles = []

for name, module, transform_net in coupling_modules:

    handles.append(
        transform_net.register_forward_hook(
            make_hook(name)
        )
    )

context_outputs = {}

flow_net.eval()

for proxy in proxies:

    captured.clear()

    ctx_data = embeddings_dict[proxy] if flow_context_dim == 128 else context_vectors_dict[proxy]

    context = (
        ctx_data
        .unsqueeze(0)
        .expand(num_probe, -1)
    )

    with torch.no_grad():
        flow_net._transform(
            y_probe,
            context,
        )

    context_outputs[proxy] = {
        name: out.clone()
        for name, out in captured.items()
    }

    print(
        f"\nProxy {proxy:+.1f}: "
        f"captured {len(captured)} conditioner outputs"
    )

print("\n" + "-" * 90)
print("CONDITIONER OUTPUT DIFFERENCES: reference = proxy -4.0")
print("-" * 90)

reference_proxy = -4.0

for proxy in proxies[1:]:

    print(f"\n-4.0 -> {proxy:+.1f}")

    common_layers = set(
        context_outputs[reference_proxy].keys()
    ).intersection(
        context_outputs[proxy].keys()
    )

    for name in sorted(common_layers):

        ref = context_outputs[reference_proxy][name]
        cur = context_outputs[proxy][name]

        delta = cur - ref

        print(
            f"  {name}: "
            f"mean|Δ|={delta.abs().mean().item():.8e}, "
            f"max|Δ|={delta.abs().max().item():.8e}, "
            f"||Δ||={torch.linalg.vector_norm(delta).item():.8e}"
        )

for handle in handles:
    handle.remove()

print("\n" + "=" * 90)
print("END TEST 5")
print("=" * 90)

S_fixed = torch.randn(num_test_samples, flow_target_dim)

for proxy in proxies:
    ctx_data = embeddings_dict[proxy] if flow_context_dim == 128 else context_vectors_dict[proxy]
    cvec = ctx_data.unsqueeze(0).expand(num_test_samples, -1)

    with torch.no_grad():
        lp = flow_net.log_prob(S_fixed, cvec)
    
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
print("\n" + "=" * 80)
print("REALISTIC FLOW-SAMPLE SENSITIVITY TEST")
print("=" * 80)

# Check the first context projection used by the flow.
expected_context_dim = 134

context_layers = []

for name, module in flow_net.named_modules():
    if isinstance(module, torch.nn.Linear):
        # Look specifically for a context_layer.
        if name.endswith("context_layer"):
            context_layers.append((name, module))

print("Detected context layers:")

for name, layer in context_layers:
    print(
        f"  {name}: "
        f"in_features={layer.in_features}, "
        f"out_features={layer.out_features}"
    )

if not context_layers:
    raise RuntimeError(
        "Could not find any context_layer in flow_net."
    )

bad_layers = [
    (name, layer.in_features)
    for name, layer in context_layers
    if layer.in_features != expected_context_dim
]

if bad_layers:
    raise RuntimeError(
        "Loaded flow is NOT a 134-D conditioned flow. "
        f"Found context dimensions: {bad_layers}"
    )

# ------------------------------------------------------------
# Draw latent/flow samples using the FULL 134-D context.
# ------------------------------------------------------------
torch.manual_seed(42)

num_test_samples = 1000

c0 = (
    context_vectors_dict[0.0]
    .unsqueeze(0)
)

print(
    "Proxy-0 context shape:",
    tuple(c0.shape)
)

assert c0.shape[-1] == 134, (
    f"Expected 134-D context, got {c0.shape[-1]}"
)

with torch.no_grad():

    S_flow_samples = flow_net.sample(
        num_test_samples,
        context=c0,
    )

# glasflow/nflows may return [1, N, 2].
if S_flow_samples.ndim == 3:
    S_flow_samples = S_flow_samples.squeeze(0)

print(
    "S_flow_samples shape:",
    tuple(S_flow_samples.shape)
)

assert S_flow_samples.shape[-1] == 2, (
    f"Expected 2D flow samples, got "
    f"{S_flow_samples.shape}"
)

# ------------------------------------------------------------
# Evaluate the SAME generated samples under every proxy
# context.
# ------------------------------------------------------------
print(
    "\n--- Sensitivity Test with S sampled from "
    "Proxy 0.0 flow ---"
)

log_probs_flow_samples = {}

for proxy in proxies:

    cvec = (
        context_vectors_dict[proxy]
        .unsqueeze(0)
        .expand(num_test_samples, -1)
    )

    assert cvec.shape[-1] == 134, (
        f"Proxy {proxy}: expected 134-D context, "
        f"got {cvec.shape[-1]}"
    )

    with torch.no_grad():

        lp = flow_net.log_prob(
            S_flow_samples,
            cvec,
        )

    log_probs_flow_samples[proxy] = lp

    print(
        f"Proxy {proxy:+.1f}: "
        f"log_prob mean={lp.mean().item():.6f}, "
        f"std={lp.std().item():.6f}, "
        f"min={lp.min().item():.6f}, "
        f"max={lp.max().item():.6f}"
    )

print(
    "\nPairwise log_prob differences "
    "on realistic flow samples:"
)

for i in range(len(proxies)):

    for j in range(i + 1, len(proxies)):

        pi = proxies[i]
        pj = proxies[j]

        lpi = log_probs_flow_samples[pi]
        lpj = log_probs_flow_samples[pj]

        diff = torch.abs(lpi - lpj)

        corr = torch.corrcoef(
            torch.stack([lpi, lpj])
        )[0, 1].item()

        print(
            f"  ({pi:+.1f}, {pj:+.1f}): "
            f"Mean |Δlp| = {diff.mean().item():.6e}, "
            f"Max |Δlp| = {diff.max().item():.6e}, "
            f"Corr = {corr:.8f}"
        )

