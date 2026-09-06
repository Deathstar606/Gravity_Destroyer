"""
LOCATE INFORMATION LOSS DIAGNOSTIC
===================================
Definitively locate where conditional information is lost in:
  waveform -> Transformer embedding -> 134D context -> DenseResidualNet -> 
  spline parameters -> NSF transformation -> posterior.

Tests:
  1. SIX-PARAMETER CONDITIONAL SIGNAL -- Pearson correlations
  2. SPLINE-PARAMETER SENSITIVITY -- per-layer spline param differences
  3. LOCALIZATION / ABLATION -- embedding vs params vs both
  4. TRAINING-TIME CHECK -- matched vs shuffled loss
  5. FINAL CLASSIFICATION

Uses the original dingo_t1.pt only for its pretrained Transformer weights,
then tests a newly initialized 2-D flow with a 134-D Beyond-GR context.
conditional information from the pretrained model is preserved.
"""

import sys
import copy
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, "/home/deathstar/dingorep/dingo")

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.core.utils.torchutils import print_number_of_model_parameters
from dingo.gw.data.event_dataset import EventDataset
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.training.train_builders import set_train_transforms

torch.manual_seed(42)
np.random.seed(42)

# =============================================================================
# SETUP
# =============================================================================
print("=" * 90)
print("LOCATE INFORMATION LOSS DIAGNOSTIC")
print("=" * 90)

device = "cpu"
if hasattr(torch, "xpu") and torch.xpu.is_available():
    device = "xpu"
elif torch.cuda.is_available():
    device = "cuda"
print(f"Device: {device}")

# Load the original checkpoint for the pretrained embedding only, then build a new 2D flow.
model_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/dingo_t1.pt"
event_path = "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/events/GW190701_203306/H1/GW190701_203306_event_data.hdf5"

print(f"\nLoading pretrained embedding from: {model_path}")
embedding_model = build_model_from_kwargs(
    filename=model_path,
    pretraining=False,
    pretrained_embedding_net=None,
    load_embedding_only=True,
    device=device,
    print_output=True,
)

with open("/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml", "r") as f:
    train_settings = yaml.safe_load(f)

model_settings = copy.deepcopy(train_settings)
model_settings["model"]["posterior_kwargs"]["input_dim"] = 2
model_settings["model"]["posterior_kwargs"]["context_dim"] = 134
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
model_2d = build_model_from_kwargs(
    settings={"dataset_settings": dataset_settings, "train_settings": model_settings},
    pretraining=False,
    pretrained_embedding_net=embedding_model.network.embedding_net,
    device=device,
    print_output=True,
)
model_2d.load_embedding_weights_only(model_path, device=device)
model = model_2d
event_dataset = EventDataset(file_name=event_path)

flow_net = model.network.flow
embedding_net = model.network.embedding_net

# Match training stage 0: freeze the embedding and train only the new flow.
embedding_net.freeze_all_except_last_n_layers(n=0)
print("\nStage-0 parameter state (matching training):")
print_number_of_model_parameters(model.network)

# Determine flow dimensions
flow_target_dim = flow_net._distribution._shape[0]
flow_context_dim = 128
for m in flow_net.modules():
    if hasattr(m, "context_features") and m.context_features is not None:
        flow_context_dim = m.context_features
        break
    if hasattr(m, "context_layer") and hasattr(m.context_layer, "in_features"):
        flow_context_dim = m.context_layer.in_features
        break

print(f"Flow target dim: {flow_target_dim}, Flow context dim: {flow_context_dim}")
assert flow_target_dim == 2, f"Expected a newly initialized 2D flow, got {flow_target_dim}D"
assert flow_context_dim == 134, f"Expected a 134D context, got {flow_context_dim}D"

# Load training data for TEST 1 and TEST 4
settings_path = "/home/deathstar/dingorep/dingo-T1/01_paper_settings/01_training/03_training/train_settings.yaml"
with open(settings_path, "r") as f:
    train_settings = yaml.safe_load(f)

wfd_path = train_settings["data"]["waveform_dataset_path"]
asd_path = train_settings["training"]["stage_0"]["asd_dataset_path"]

print(f"\nLoading WaveformDataset from: {wfd_path}")
wfd = WaveformDataset(file_name=wfd_path)

print("Setting up training transforms...")
set_train_transforms(
    wfd=wfd,
    data_settings=train_settings["data"],
    asd_dataset_path=asd_path,
    print_output=False,
)

from torch.utils.data import DataLoader
loader = DataLoader(wfd, batch_size=256, shuffle=False, num_workers=0)

print("Extracting precomputed batch...")
with torch.no_grad():
    raw_batch = next(iter(loader))

# batch: [theta, waveform, context_params, position, padding_mask]
theta_batch = raw_batch[0].float()         # [B, 2] = [beta_residual, chirp_mass]
wf_batch = raw_batch[1].float()            # [B, N, 48]
ctx_params_batch = raw_batch[2].float()    # [B, 6] = [beta_proxy, q, chi1z, chi2z, dL, theta_jn]
pos_batch = raw_batch[3].float()           # [B, N, 3]
mask_batch = raw_batch[4].bool()           # [B, N]

B = theta_batch.shape[0]
print(f"Batch size: {B}")
print(f"theta: {theta_batch.shape}, wf: {wf_batch.shape}, ctx_params: {ctx_params_batch.shape}")
print(f"pos: {pos_batch.shape}, mask: {mask_batch.shape}")

# Move all batch tensors to device
theta_batch = theta_batch.to(device)
wf_batch = wf_batch.to(device)
ctx_params_batch = ctx_params_batch.to(device)
pos_batch = pos_batch.to(device)
mask_batch = mask_batch.to(device)

print(f"\nTensors moved to device: {device}")

# Compute embeddings
embedding_net.eval()
with torch.no_grad():
    emb_batch = embedding_net(wf_batch, pos_batch, mask_batch)
    if isinstance(emb_batch, tuple):
        emb_batch = emb_batch[0]

print(f"Embedding: {emb_batch.shape}")

# Full 134-D context = [embedding(128) || context_params(6)]
full_context_batch = torch.cat([emb_batch, ctx_params_batch], dim=-1)  # [B, 134]
print(f"Full context (for reference): {full_context_batch.shape}")

# ============================================================================
# TEST 1: SIX-PARAMETER CONDITIONAL SIGNAL
# ============================================================================
print("\n" + "=" * 90)
print("TEST 1: SIX-PARAMETER CONDITIONAL SIGNAL")
print("=" * 90)
print("Pearson correlations between conditioning parameters and targets")
print("Conditioning params: beta_proxy, mass_ratio, chi1z, chi2z, luminosity_distance, theta_jn")
print("Targets: beta_residual, chirp_mass")

param_names = ["beta_proxy", "mass_ratio", "chi1z", "chi2z", "luminosity_distance", "theta_jn"]
target_names = ["beta_residual", "chirp_mass"]

ctx_np = ctx_params_batch.cpu().numpy()  # [B, 6]
theta_np = theta_batch.cpu().numpy()     # [B, 2]

print(f"\n{'Parameter':<22} | {'vs beta_residual':>17} | {'vs chirp_mass':>14}")
print("-" * 60)

corr_matrix = np.zeros((6, 2))
for i, pname in enumerate(param_names):
    for j, tname in enumerate(target_names):
        r = np.corrcoef(ctx_np[:, i], theta_np[:, j])[0, 1]
        corr_matrix[i, j] = r
    print(f"{pname:<22} | {corr_matrix[i, 0]:>+17.6f} | {corr_matrix[i, 1]:>+14.6f}")

# Also correlate embedding dimensions with targets
emb_np = emb_batch.cpu().numpy()  # [B, 128]
emb_target_corr = np.zeros((128, 2))
for i in range(128):
    for j in range(2):
        emb_target_corr[i, j] = np.corrcoef(emb_np[:, i], theta_np[:, j])[0, 1]

top_k = 10
print(f"\nTop-{top_k} embedding dimensions most correlated with beta_residual:")
br_corr_abs = np.abs(emb_target_corr[:, 0])
top_br = np.argsort(br_corr_abs)[::-1][:top_k]
for rank, idx in enumerate(top_br):
    print(f"  dim {idx:3d}: r = {emb_target_corr[idx, 0]:+.6f} (|r| = {br_corr_abs[idx]:.6f})")

print(f"\nTop-{top_k} embedding dimensions most correlated with chirp_mass:")
cm_corr_abs = np.abs(emb_target_corr[:, 1])
top_cm = np.argsort(cm_corr_abs)[::-1][:top_k]
for rank, idx in enumerate(top_cm):
    print(f"  dim {idx:3d}: r = {emb_target_corr[idx, 1]:+.6f} (|r| = {cm_corr_abs[idx]:.6f})")

# Summary stats
print(f"\nEmbedding-target correlation summary:")
print(f"  beta_residual: max|r| = {br_corr_abs.max():.6f}, mean|r| = {br_corr_abs.mean():.6f}, median|r| = {np.median(br_corr_abs):.6f}")
print(f"  chirp_mass:    max|r| = {cm_corr_abs.max():.6f}, mean|r| = {cm_corr_abs.mean():.6f}, median|r| = {np.median(cm_corr_abs):.6f}")

# ============================================================================
# TEST 2: SPLINE-PARAMETER SENSITIVITY
# ============================================================================
print("\n" + "=" * 90)
print("TEST 2: SPLINE-PARAMETER SENSITIVITY")
print("=" * 90)
print("For each coupling layer, compare spline parameters under different contexts")

# Use real data: pick two samples with extreme beta_proxy values
beta_proxy_vals = ctx_params_batch[:, 0].cpu().numpy()
idx_low = int(np.argmin(beta_proxy_vals))
idx_high = int(np.argmax(beta_proxy_vals))
print(f"Low beta_proxy sample: idx={idx_low}, beta_proxy={beta_proxy_vals[idx_low]:.4f}")
print(f"High beta_proxy sample: idx={idx_high}, beta_proxy={beta_proxy_vals[idx_high]:.4f}")

# The new flow receives the 134D vector [embedding(128) || parameters(6)].
ctx_low = torch.cat([emb_batch[idx_low:idx_low+1], ctx_params_batch[idx_low:idx_low+1]], dim=-1)
ctx_high = torch.cat([emb_batch[idx_high:idx_high+1], ctx_params_batch[idx_high:idx_high+1]], dim=-1)

print(f"Context low shape: {ctx_low.shape}, norm: {ctx_low.norm().item():.4f}")
print(f"Context high shape: {ctx_high.shape}, norm: {ctx_high.norm().item():.4f}")
print(f"||ctx_high - ctx_low||: {(ctx_high - ctx_low).norm().item():.4f}")

# Fixed target samples
torch.manual_seed(123)
N_probe = 512
y_fixed = torch.randn(N_probe, flow_target_dim).to(device)

# Find all coupling transforms
coupling_layers = []
for name, mod in flow_net.named_modules():
    if hasattr(mod, "transform_net") and hasattr(mod, "identity_features"):
        coupling_layers.append((name, mod))

print(f"\nFound {len(coupling_layers)} coupling layers")

# Determine num_bins from first coupling layer
num_bins = coupling_layers[0][1].num_bins if hasattr(coupling_layers[0][1], 'num_bins') else 8
print(f"num_bins = {num_bins}")

# Hook into each coupling layer's transform_net to capture spline params
spline_params_low = {}
spline_params_high = {}

def make_capture_hook(store, layer_name):
    def hook(module, inputs, output):
        store[layer_name] = output.detach().cpu().clone()
    return hook

# Run forward with LOW context
handles = []
for name, mod in coupling_layers:
    handles.append(mod.transform_net.register_forward_hook(make_capture_hook(spline_params_low, name)))

ctx_low_expanded = ctx_low.expand(N_probe, -1)
flow_net.eval()
with torch.no_grad():
    z_low, logdet_low = flow_net._transform(y_fixed, ctx_low_expanded)
for h in handles:
    h.remove()

# Run forward with HIGH context
handles = []
for name, mod in coupling_layers:
    handles.append(mod.transform_net.register_forward_hook(make_capture_hook(spline_params_high, name)))

ctx_high_expanded = ctx_high.expand(N_probe, -1)
with torch.no_grad():
    z_high, logdet_high = flow_net._transform(y_fixed, ctx_high_expanded)
for h in handles:
    h.remove()

print(f"\n{'Layer':<50} | {'mean|D|':>10} | {'max|D|':>10} | {'relL2':>10} | {'mean|Dw|':>10} | {'mean|Dh|':>10} | {'mean|Dd|':>10}")
print("-" * 130)

for name in sorted(spline_params_low.keys()):
    sp_lo = spline_params_low[name]
    sp_hi = spline_params_high[name]

    delta = sp_hi - sp_lo
    mean_abs = delta.abs().mean().item()
    max_abs = delta.abs().max().item()
    rel_l2 = delta.norm().item() / (sp_lo.norm().item() + 1e-12)

    # Decompose into widths, heights, derivatives per transform feature
    n_tf = sp_lo.shape[1]
    params_per_feature = 3 * num_bins - 1
    n_features = n_tf // params_per_feature if params_per_feature > 0 else 1

    if n_features > 0 and n_tf == n_features * params_per_feature:
        sp_lo_r = sp_lo.reshape(N_probe, n_features, params_per_feature)
        sp_hi_r = sp_hi.reshape(N_probe, n_features, params_per_feature)
        delta_r = sp_hi_r - sp_lo_r

        w_delta = delta_r[..., :num_bins].abs().mean().item()
        h_delta = delta_r[..., num_bins:2*num_bins].abs().mean().item()
        d_delta = delta_r[..., 2*num_bins:].abs().mean().item()
    else:
        w_delta = h_delta = d_delta = float('nan')

    print(f"{name:<50} | {mean_abs:>10.6f} | {max_abs:>10.6f} | {rel_l2:>10.6f} | {w_delta:>10.6f} | {h_delta:>10.6f} | {d_delta:>10.6f}")

# Compare final transformed output and logdet
dz = z_high.cpu() - z_low.cpu()
dlogdet = logdet_high.cpu() - logdet_low.cpu()
print(f"\nFinal transformation comparison (low vs high beta_proxy context):")
print(f"  z_low:  mean={z_low.mean().item():.6f}, std={z_low.std().item():.6f}")
print(f"  z_high: mean={z_high.mean().item():.6f}, std={z_high.std().item():.6f}")
print(f"  mean|Dz|     = {dz.abs().mean().item():.8e}")
print(f"  max|Dz|      = {dz.abs().max().item():.8e}")
print(f"  ||Dz||       = {dz.norm().item():.8e}")
print(f"  relL2(z)     = {dz.norm().item() / (z_low.cpu().norm().item() + 1e-12):.8e}")
print(f"  mean|Dlogdet| = {dlogdet.abs().mean().item():.8e}")
print(f"  max|Dlogdet|  = {dlogdet.abs().max().item():.8e}")

# Log-prob comparison
with torch.no_grad():
    lp_low = flow_net.log_prob(y_fixed, ctx_low_expanded).cpu()
    lp_high = flow_net.log_prob(y_fixed, ctx_high_expanded).cpu()

dlp = lp_high - lp_low
print(f"\n  log_prob low:  mean={lp_low.mean().item():.6f}, std={lp_low.std().item():.6f}")
print(f"  log_prob high: mean={lp_high.mean().item():.6f}, std={lp_high.std().item():.6f}")
print(f"  mean|Dlp|    = {dlp.abs().mean().item():.8e}")
print(f"  max|Dlp|     = {dlp.abs().max().item():.8e}")
print(f"  correlation  = {torch.corrcoef(torch.stack([lp_low.flatten(), lp_high.flatten()]))[0,1].item():.8f}")

# ============================================================================
# TEST 3: LOCALIZATION / ABLATION
# ============================================================================
print("\n" + "=" * 90)
print("TEST 3: LOCALIZATION / ABLATION")
print("=" * 90)
print("Using the SAME fixed targets, compare:")
print("  A) Change ONLY embedding (keep same params)")
print("  B) Change ONLY 6-D parameters (keep same embedding)")
print("  C) Change both embedding and parameters")

emb_low = emb_batch[idx_low:idx_low+1]
emb_high = emb_batch[idx_high:idx_high+1]
params_low = ctx_params_batch[idx_low:idx_low+1]
params_high = ctx_params_batch[idx_high:idx_high+1]

print(f"\nReference (low):  beta_proxy={params_low[0,0].item():.4f}")
print(f"Comparison (high): beta_proxy={params_high[0,0].item():.4f}")

params_low = ctx_params_batch[idx_low:idx_low+1]
params_high = ctx_params_batch[idx_high:idx_high+1]
ctx_ref = torch.cat([emb_low, params_low], dim=-1).expand(N_probe, -1)
ctx_A = torch.cat([emb_high, params_low], dim=-1).expand(N_probe, -1)
ctx_B = torch.cat([emb_low, params_high], dim=-1).expand(N_probe, -1)
ctx_C = torch.cat([emb_high, params_high], dim=-1).expand(N_probe, -1)

with torch.no_grad():
    z_ref, logdet_ref = flow_net._transform(y_fixed, ctx_ref)
    z_A, logdet_A = flow_net._transform(y_fixed, ctx_A)
    z_B, logdet_B = flow_net._transform(y_fixed, ctx_B)
    z_C, logdet_C = flow_net._transform(y_fixed, ctx_C)

    lp_ref = flow_net.log_prob(y_fixed, ctx_ref)
    lp_A = flow_net.log_prob(y_fixed, ctx_A)
    lp_B = flow_net.log_prob(y_fixed, ctx_B)
    lp_C = flow_net.log_prob(y_fixed, ctx_C)

dz_A = (z_A.cpu() - z_ref.cpu())
dlp_A = (lp_A.cpu() - lp_ref.cpu())
dz_B = (z_B.cpu() - z_ref.cpu())
dlp_B = (lp_B.cpu() - lp_ref.cpu())
dz_C = (z_C.cpu() - z_ref.cpu())
dlp_C = (lp_C.cpu() - lp_ref.cpu())

print(f"\n{'Condition':<30} | {'mean|Dz|':>12} | {'max|Dz|':>12} | {'mean|Dlp|':>12} | {'max|Dlp|':>12}")
print("-" * 90)
print(f"{'A: Change embedding only':<30} | {dz_A.abs().mean().item():>12.8f} | {dz_A.abs().max().item():>12.8f} | {dlp_A.abs().mean().item():>12.8f} | {dlp_A.abs().max().item():>12.8f}")
print(f"{'B: Change parameters only':<30} | {dz_B.abs().mean().item():>12.8f} | {dz_B.abs().max().item():>12.8f} | {dlp_B.abs().mean().item():>12.8f} | {dlp_B.abs().max().item():>12.8f}")
print(f"{'C: Change both':<30} | {dz_C.abs().mean().item():>12.8f} | {dz_C.abs().max().item():>12.8f} | {dlp_C.abs().mean().item():>12.8f} | {dlp_C.abs().max().item():>12.8f}")

# Multi-sample embedding sensitivity
print(f"\n--- Multi-sample embedding sensitivity (10 random pairs) ---")
np.random.seed(42)
pair_indices = [(np.random.randint(B), np.random.randint(B)) for _ in range(10)]

mean_dlp_list = []
mean_dz_list = []
emb_dist_list = []

for i_a, i_b in pair_indices:
    c_a = torch.cat([emb_batch[i_a:i_a+1], ctx_params_batch[i_a:i_a+1]], dim=-1).expand(N_probe, -1)
    c_b = torch.cat([emb_batch[i_b:i_b+1], ctx_params_batch[i_b:i_b+1]], dim=-1).expand(N_probe, -1)

    with torch.no_grad():
        lp_a = flow_net.log_prob(y_fixed, c_a).cpu()
        lp_b = flow_net.log_prob(y_fixed, c_b).cpu()
        z_a, _ = flow_net._transform(y_fixed, c_a)
        z_b, _ = flow_net._transform(y_fixed, c_b)

    emb_dist = (emb_batch[i_a] - emb_batch[i_b]).norm().item()
    mean_dlp = (lp_a - lp_b).abs().mean().item()
    mean_dz_val = (z_a.cpu() - z_b.cpu()).abs().mean().item()

    mean_dlp_list.append(mean_dlp)
    mean_dz_list.append(mean_dz_val)
    emb_dist_list.append(emb_dist)

    bp_a = ctx_params_batch[i_a, 0].item()
    bp_b = ctx_params_batch[i_b, 0].item()
    print(f"  Pair ({i_a},{i_b}): ||Demb||={emb_dist:.4f}, Dbeta_proxy={bp_a-bp_b:+.4f}, mean|Dlp|={mean_dlp:.6f}, mean|Dz|={mean_dz_val:.6f}")

# Correlation between embedding distance and output sensitivity
if len(emb_dist_list) > 2:
    from scipy import stats as spstats
    r_dlp, p_dlp = spstats.pearsonr(emb_dist_list, mean_dlp_list)
    r_dz, p_dz = spstats.pearsonr(emb_dist_list, mean_dz_list)
    print(f"\n  Corr(||Demb||, mean|Dlp|) = {r_dlp:.4f} (p={p_dlp:.4f})")
    print(f"  Corr(||Demb||, mean|Dz|)  = {r_dz:.4f} (p={p_dz:.4f})")

# ============================================================================
# TEST 4: TRAINING-TIME CHECK
# ============================================================================
print("\n" + "=" * 90)
print("TEST 4: TRAINING-TIME CHECK")
print("=" * 90)
print("Using real training batch: matched target/context vs shuffled")

print("\nNote: Using 134-D [embedding || six parameters] context")
print(f"      Using the real {flow_target_dim}-D target batch")

torch.manual_seed(42)
N_check = min(B, 256)
emb_check = emb_batch[:N_check]
params_check = ctx_params_batch[:N_check]
context_check = torch.cat([emb_check, params_check], dim=-1)
theta_check = theta_batch[:N_check]

# Matched: theta_i with context_i
with torch.no_grad():
    lp_matched = flow_net.log_prob(theta_check, context_check).cpu()

# Shuffled full context
perm = torch.randperm(N_check)
context_shuffled = context_check[perm]
with torch.no_grad():
    lp_shuffled_full = flow_net.log_prob(theta_check, context_shuffled).cpu()

# Shuffle waveforms by recomputing the embedding, while keeping each sample's parameters.
with torch.no_grad():
    shuffled_emb = embedding_net(wf_batch[:N_check][perm], pos_batch[:N_check][perm], mask_batch[:N_check][perm])
    if isinstance(shuffled_emb, tuple):
        shuffled_emb = shuffled_emb[0]
context_shuffled_waveform = torch.cat([shuffled_emb, params_check], dim=-1)
with torch.no_grad():
    lp_shuffled_waveform = flow_net.log_prob(theta_check, context_shuffled_waveform).cpu()

# Shuffle beta_proxy only, leaving the other five parameters and embedding matched.
params_beta_shuffled = params_check.clone()
params_beta_shuffled[:, 0] = params_check[perm, 0]
context_beta_shuffled = torch.cat([emb_check, params_beta_shuffled], dim=-1)
with torch.no_grad():
    lp_shuffled_beta = flow_net.log_prob(theta_check, context_beta_shuffled).cpu()

matched_loss = -lp_matched.mean().item()
shuffled_full_loss = -lp_shuffled_full.mean().item()
shuffled_waveform_loss = -lp_shuffled_waveform.mean().item()
shuffled_beta_loss = -lp_shuffled_beta.mean().item()

print(f"\n{'Condition':<40} | {'Loss (-log_prob)':>16} | {'D vs matched':>14}")
print("-" * 80)
print(f"{'Matched (theta_i, context_i)':<40} | {matched_loss:>16.6f} | {'---':>14}")
print(f"{'Shuffled full context':<40} | {shuffled_full_loss:>16.6f} | {shuffled_full_loss - matched_loss:>+14.6f}")
print(f"{'Shuffled waveform':<40} | {shuffled_waveform_loss:>16.6f} | {shuffled_waveform_loss - matched_loss:>+14.6f}")
print(f"{'Shuffled beta_proxy only':<40} | {shuffled_beta_loss:>16.6f} | {shuffled_beta_loss - matched_loss:>+14.6f}")

# Linear probe: can we predict targets from embedding?
print(f"\n--- Embedding -> target regression check (linear probe) ---")
from torch.linalg import lstsq

X = emb_batch.cpu().float()
Y = theta_batch.cpu().float()
X_bias = torch.cat([X, torch.ones(B, 1)], dim=1)
W = lstsq(X_bias, Y).solution
Y_pred = X_bias @ W
residuals = Y - Y_pred
r2_beta = 1 - (residuals[:, 0].var() / Y[:, 0].var()).item()
r2_chirp = 1 - (residuals[:, 1].var() / Y[:, 1].var()).item()

print(f"  Linear probe R2 for beta_residual from 128-D embedding: {r2_beta:.6f}")
print(f"  Linear probe R2 for chirp_mass from 128-D embedding:    {r2_chirp:.6f}")

# Same with only 6 context params
X_params = ctx_params_batch.cpu().float()
X_params_bias = torch.cat([X_params, torch.ones(B, 1)], dim=1)
W_params = lstsq(X_params_bias, Y).solution
Y_pred_params = X_params_bias @ W_params
res_params = Y - Y_pred_params
r2_beta_params = 1 - (res_params[:, 0].var() / Y[:, 0].var()).item()
r2_chirp_params = 1 - (res_params[:, 1].var() / Y[:, 1].var()).item()

print(f"  Linear probe R2 for beta_residual from 6 context params: {r2_beta_params:.6f}")
print(f"  Linear probe R2 for chirp_mass from 6 context params:    {r2_chirp_params:.6f}")

# Combined
X_full = torch.cat([X, X_params], dim=1)
X_full_bias = torch.cat([X_full, torch.ones(B, 1)], dim=1)
W_full = lstsq(X_full_bias, Y).solution
Y_pred_full = X_full_bias @ W_full
res_full = Y - Y_pred_full
r2_beta_full = 1 - (res_full[:, 0].var() / Y[:, 0].var()).item()
r2_chirp_full = 1 - (res_full[:, 1].var() / Y[:, 1].var()).item()

print(f"  Linear probe R2 for beta_residual from 134-D combined:   {r2_beta_full:.6f}")
print(f"  Linear probe R2 for chirp_mass from 134-D combined:      {r2_chirp_full:.6f}")

# ============================================================================
# TEST 5: GRADIENT FLOW THROUGH DenseResidualNet
# ============================================================================
print("\n" + "=" * 90)
print("TEST 5: GRADIENT FLOW THROUGH DenseResidualNet")
print("=" * 90)
print("Per-layer gradient magnitude from log_prob -> context")

ctx_grad = torch.cat([emb_batch[0:1], ctx_params_batch[0:1]], dim=-1).detach().expand(N_probe, -1).clone().requires_grad_(True)
theta_probe = torch.randn(N_probe, flow_target_dim).to(device)

lp_grad = flow_net.log_prob(theta_probe, ctx_grad)
scalar = lp_grad.mean()

grad_context = torch.autograd.grad(scalar, ctx_grad, retain_graph=False)[0]
print(f"||dL/d_context|| = {grad_context.norm().item():.8e}")
print(f"mean|dL/d_context| = {grad_context.abs().mean().item():.8e}")
print(f"max|dL/d_context| = {grad_context.abs().max().item():.8e}")

# Per-dimension gradient
per_dim_grad = grad_context.abs().mean(dim=0)
print(f"\nPer-dimension gradient magnitude (mean across samples):")
print(f"  min dim grad: {per_dim_grad.min().item():.8e}")
print(f"  max dim grad: {per_dim_grad.max().item():.8e}")
print(f"  mean dim grad: {per_dim_grad.mean().item():.8e}")
print(f"  std dim grad: {per_dim_grad.std().item():.8e}")

# Top-10 most sensitive context dimensions
top_grad_dims = torch.argsort(per_dim_grad, descending=True)[:10]
print(f"\n  Top-10 most gradient-sensitive context dims:")
for rank, d in enumerate(top_grad_dims):
    print(f"    dim {d.item():3d}: |grad| = {per_dim_grad[d].item():.8e}")

# ============================================================================
# FINAL CLASSIFICATION
# ============================================================================
print("\n" + "=" * 90)
print("FINAL CLASSIFICATION: WHERE IS CONDITIONAL INFORMATION LOST?")
print("=" * 90)

# Collect evidence
print("\n--- Evidence Summary ---\n")

# T1: Correlations
max_emb_corr_br = br_corr_abs.max()
max_emb_corr_cm = cm_corr_abs.max()
max_param_corr_br = np.abs(corr_matrix[:, 0]).max()
max_param_corr_cm = np.abs(corr_matrix[:, 1]).max()

print(f"T1 - Parameter Correlations:")
print(f"  max|r(param, beta_res)| = {max_param_corr_br:.4f}")
print(f"  max|r(param, chirp_mass)| = {max_param_corr_cm:.4f}")
print(f"  max|r(emb_dim, beta_res)| = {max_emb_corr_br:.4f}")
print(f"  max|r(emb_dim, chirp_mass)| = {max_emb_corr_cm:.4f}")

# T2: Spline sensitivity
all_mean_deltas = [spline_params_high[k].sub(spline_params_low[k]).abs().mean().item() 
                   for k in spline_params_low]
mean_spline_delta = np.mean(all_mean_deltas) if all_mean_deltas else 0
max_spline_delta = np.max(all_mean_deltas) if all_mean_deltas else 0

print(f"\nT2 - Spline Parameter Sensitivity:")
print(f"  mean of mean|Dspline| across layers = {mean_spline_delta:.6f}")
print(f"  max of mean|Dspline| across layers = {max_spline_delta:.6f}")
print(f"  mean|Dz| (final transform) = {dz.abs().mean().item():.6f}")
print(f"  mean|Dlog_prob| = {dlp.abs().mean().item():.6f}")

# T3: Ablation
print(f"\nT3 - Localization/Ablation:")
print(f"  A: Change embedding -> mean|Dlp| = {dlp_A.abs().mean().item():.6f}")
print(f"  B: Change parameters -> mean|Dlp| = {dlp_B.abs().mean().item():.6f}")
print(f"  C: Change both -> mean|Dlp| = {dlp_C.abs().mean().item():.6f}")

# T4: Training check
print(f"\nT4 - Training-time check:")
print(f"  Dloss (shuffled_full - matched) = {shuffled_full_loss - matched_loss:+.6f}")
print(f"  Dloss (shuffled_waveform - matched) = {shuffled_waveform_loss - matched_loss:+.6f}")
print(f"  Dloss (shuffled_beta - matched) = {shuffled_beta_loss - matched_loss:+.6f}")
print(f"  R2(emb->beta_res) = {r2_beta:.4f}, R2(emb->chirp_mass) = {r2_chirp:.4f}")
print(f"  R2(params->beta_res) = {r2_beta_params:.4f}, R2(params->chirp_mass) = {r2_chirp_params:.4f}")

# T5: Gradients
print(f"\nT5 - Gradient flow:")
print(f"  mean|dL/d_context| = {grad_context.abs().mean().item():.8e}")

# Classification logic
print("\n" + "-" * 90)
print("CLASSIFICATION:")
print("-" * 90)

stages = {
    "A": "Before Transformer (data generation / rotation)",
    "B": "Transformer representation (128-D embedding)",
    "C": "134-D context construction",
    "D": "DenseResidualNet / context pathway (GLU gating)",
    "E": "Spline parameter generation",
    "F": "Final NSF transformation",
    "G": "Nowhere - conditional information is preserved",
}

# Decision tree
if max_param_corr_br < 0.05 and max_emb_corr_br < 0.05:
    verdict = "A"
    explanation = "No correlation found between conditioning parameters/embedding and targets in training data."
elif max_emb_corr_br < 0.05 and max_param_corr_br > 0.1:
    verdict = "B"
    explanation = "Parameters carry signal but embedding does not -- Transformer fails to encode it."
elif r2_beta < 0.01 and max_emb_corr_br > 0.05:
    verdict = "B"
    explanation = "Individual embedding dims correlate weakly but linear combination fails -- representation is noisy."
elif mean_spline_delta < 1e-4 and grad_context.abs().mean().item() < 1e-6:
    verdict = "D"
    explanation = "Spline params barely change and gradient is near-zero -- DenseResidualNet/GLU pathway ignores context."
elif mean_spline_delta < 1e-3:
    verdict = "E"
    explanation = "Spline parameters show minimal sensitivity to context changes."
elif dz.abs().mean().item() < 0.01 and mean_spline_delta > 0.01:
    verdict = "F"
    explanation = "Spline params differ but final transformation is similar -- spline changes cancel out."
elif (shuffled_full_loss - matched_loss) > 0.5 and dlp.abs().mean().item() > 1.0:
    verdict = "G"
    explanation = "Conditional information IS preserved -- shuffling hurts loss, context changes outputs."
else:
    verdict = "MIXED"
    explanation = (
        f"Evidence is mixed. Spline D={mean_spline_delta:.4f}, "
        f"z D={dz.abs().mean().item():.4f}, "
        f"lp D={dlp.abs().mean().item():.4f}, "
        f"shuffle Dloss={shuffled_full_loss - matched_loss:.4f}, "
        f"R2(emb->beta)={r2_beta:.4f}. "
        f"Check if signal is present but weak, or if it is lost at a specific stage."
    )

print(f"\n  VERDICT: Stage {verdict}")
print(f"  -> {stages.get(verdict, verdict)}")
print(f"  EXPLANATION: {explanation}")

print("\n" + "=" * 90)
print("DIAGNOSTIC COMPLETE")
print("=" * 90)
