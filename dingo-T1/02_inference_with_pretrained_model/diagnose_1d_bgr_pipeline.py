#!/usr/bin/env python3
"""
Standalone Diagnostic Script for 1D Beyond-GR Inference Pipeline.
==================================================================
Model Architecture:
  waveform -> preprocessing/tokens -> Transformer embedding [B, 128]
           -> 1D Neural Spline Flow [B, 1] -> p(beta_true | embedding)

Verifications:
  1. Pipeline / Tensor-Shape Validation:
     - Input waveform
     - Waveform preprocessing
     - Transformer embedding [B, 128]
     - Flow context [B, 128]
     - Raw 1D NSF sample output [N, 1]
     - beta_true extraction (de-standardization)
     - Posterior samples returned by BeyondGRSampler [N,]
     - Final NumPy / DataFrame representation
     - Strict audit: NO legacy parameters (beta_proxy, beta_residual, chirp_mass, tilts, etc.)
  2. Injected beta0 Recovery Test:
     - Evaluates across held-out validation waveforms with known true beta0 in [-5, 5]
     - Computes posterior median, 90% credible interval (5th-95th percentile), width, bias, coverage
  3. Recovery & Bias Plots:
     - beta_recovery.png: recovered median vs injected beta0 with y=x reference line and 90% CI error bars
     - beta_bias.png: bias (median - true) vs injected beta0 with y=0 reference line
  4. Quantitative Report & Scientific Sanity Checks:
     - Pearson and Spearman correlations
     - Empirical vs nominal coverage
     - Posterior collapse check (variance ratio)
     - Boundary saturation and systematic bias analysis
  5. Short PASS/FAIL summary across all 6 verification areas.
"""

import sys
import os
import copy
import argparse
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Ensure dingo package is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
dingo_pkg = os.path.join(repo_root, "dingo")
if dingo_pkg not in sys.path:
    sys.path.insert(0, dingo_pkg)

from dingo.core.posterior_models.build_model import build_model_from_kwargs
from dingo.gw.inference.gw_samplers import create_sampler, BeyondGRSampler
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.noise.asd_dataset import ASDDataset
from bilby.gw.detector import InterferometerList
from dingo.gw.transforms.detector_transforms import ProjectOntoDetectors, GetDetectorTimes
from dingo.gw.transforms.noise_transforms import SampleNoiseASD
from dingo.gw.transforms.parameter_transforms import SampleExtrinsicParameters
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.core.utils.torchutils import split_dataset_into_train_and_test

PROHIBITED_KEYS = {
    "beta_proxy",
    "beta_residual",
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "phase",
    "luminosity_distance",
    "geocent_time",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="1D Beyond-GR Inference Pipeline Diagnostics & Recovery Suite"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_latest.pt",
        help="Path to trained 1D Beyond-GR model checkpoint (default: model_latest.pt)",
    )
    parser.add_argument(
        "--wfd_path",
        type=str,
        default=None,
        help="Path to waveform dataset HDF5 (default: read from model metadata)",
    )
    parser.add_argument(
        "--asd_path",
        type=str,
        default=None,
        help="Path to ASD dataset HDF5 (default: read from model metadata)",
    )
    parser.add_argument(
        "--num_val_waveforms",
        type=int,
        default=50,
        help="Number of validation waveforms to evaluate (default: 50)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of posterior samples per validation waveform (default: 2000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Batch size for posterior sampling (default: 2000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "xpu", "cuda"],
        help="Device to run inference on (default: cpu)",
    )
    parser.add_argument(
        "--zero_noise",
        action="store_true",
        default=False,
        help="If set, evaluate in zero-noise limit (default: False, realistic noise added)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diagnostics",
        help="Directory to save diagnostic outputs (default: diagnostics)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


# ==============================================================================
# 1. PIPELINE / TENSOR-SHAPE VALIDATION
# ==============================================================================
def run_pipeline_tensor_validation(model, sampler, test_sample, args):
    """
    Traces a single event through the entire inference pipeline, validating
    tensor shapes and confirming the strict absence of legacy parameters.
    """
    print("\n" + "=" * 80)
    print("STAGE 1: PIPELINE / TENSOR-SHAPE VALIDATION")
    print("=" * 80)

    N = 100  # Verification sample count
    context = {"waveform": test_sample["waveform"], "asds": test_sample["asds"]}

    # --- 1.1 Input waveform ---
    print("\n[1.1 Input Waveform & ASDs]")
    for ifo in sampler.detectors:
        wf = context["waveform"][ifo]
        asd = context["asds"][ifo]
        print(f"  IFO {ifo}: waveform type={type(wf).__name__}, shape={wf.shape}, dtype={wf.dtype}")
        print(f"  IFO {ifo}: ASD      type={type(asd).__name__}, shape={asd.shape}, dtype={asd.dtype}")
        if not np.all(np.isfinite(wf)):
            raise AssertionError(f"FATAL: Waveform for {ifo} contains non-finite values!")
        if not np.all(np.isfinite(asd)):
            raise AssertionError(f"FATAL: ASD for {ifo} contains non-finite values!")

    # --- 1.2 Waveform Preprocessing ---
    print("\n[1.2 Waveform Preprocessing (sampler.transform_pre)]")
    x = context.copy()
    for i, transform in enumerate(sampler.transform_pre.transforms):
        t_name = type(transform).__name__
        x = transform(x)
        if isinstance(x, dict):
            summary = f"dict with keys={list(x.keys())}"
        elif isinstance(x, (list, tuple)):
            shapes = [getattr(elem, "shape", type(elem)) for elem in x]
            summary = f"{type(x).__name__} len={len(x)} shapes={shapes}"
        elif hasattr(x, "shape"):
            summary = f"tensor shape={x.shape}, dtype={x.dtype}"
        else:
            summary = f"type={type(x).__name__}"
        print(f"  Transform {i:2d} ({t_name:30s}) -> {summary}")

    # After transform_pre, x is a list/tuple: (waveform_tokens, position, drop_token_mask)
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        raise AssertionError(f"Expected 3 tokenized tensors from transform_pre, got {type(x)} with len {len(x)}")

    wf_tokens, pos_tokens, mask_tokens = x
    # Ensure batch dimension
    if wf_tokens.ndim == 2:
        wf_tokens = wf_tokens.unsqueeze(0)
    if pos_tokens.ndim == 2:
        pos_tokens = pos_tokens.unsqueeze(0)
    if mask_tokens.ndim == 1:
        mask_tokens = mask_tokens.unsqueeze(0)

    batch_size = wf_tokens.shape[0]
    print(f"\n  Batch input shapes to embedding network:")
    print(f"    waveform tokens : shape={wf_tokens.shape} (expected: (batch, num_tokens, token_channels))")
    print(f"    positions       : shape={pos_tokens.shape}")
    print(f"    mask            : shape={mask_tokens.shape}")

    # --- 1.3 Transformer Embedding ---
    print("\n[1.3 Transformer Embedding]")
    model.network.eval()
    with torch.no_grad():
        embedding = model.network.embedding_net(wf_tokens, pos_tokens, mask_tokens)
        if isinstance(embedding, tuple):
            embedding = embedding[0]

    print(f"  Transformer output embedding shape: {embedding.shape}")
    if embedding.shape != (batch_size, 128):
        raise AssertionError(f"FATAL: Expected embedding shape ({batch_size}, 128), got {embedding.shape}!")
    print("  --> VERIFIED: Transformer embedding is exactly (batch, 128)")

    # --- 1.4 Flow Context ---
    print("\n[1.4 Embedding Passed to Flow]")
    # In 1D Beyond-GR, context_dim == 128 with no appended parameters
    flow_context = embedding
    print(f"  Flow conditioner context shape: {flow_context.shape}")
    if flow_context.shape != (batch_size, 128):
        raise AssertionError(f"FATAL: Expected flow context shape ({batch_size}, 128), got {flow_context.shape}!")
    print("  --> VERIFIED: Flow context is exactly (batch, 128) with no extra parameters")

    # --- 1.5 Raw NSF Sample Output ---
    print(f"\n[1.5 Raw NSF Sample Output (N={N} samples)]")
    with torch.no_grad():
        y_raw, log_prob_raw = model.network.flow.sample_and_log_prob(
            num_samples=N, context=flow_context
        )
    print(f"  Raw flow sample y shape        : {y_raw.shape}")
    print(f"  Raw flow log_prob shape        : {log_prob_raw.shape}")

    # Squeeze event batch dimension (axis 0) if batch_size == 1
    if y_raw.ndim == 3 and y_raw.shape[0] == 1:
        y_squeezed = y_raw.squeeze(0)
    else:
        y_squeezed = y_raw

    if log_prob_raw.ndim == 2 and log_prob_raw.shape[0] == 1:
        log_prob_squeezed = log_prob_raw.squeeze(0)
    else:
        log_prob_squeezed = log_prob_raw

    print(f"  Event-squeezed flow y shape    : {y_squeezed.shape} (expected: ({N}, 1))")
    print(f"  Event-squeezed log_prob shape  : {log_prob_squeezed.shape} (expected: ({N},))")

    if y_squeezed.shape != (N, 1):
        raise AssertionError(f"FATAL: Raw 1D flow output shape is {y_squeezed.shape}, expected ({N}, 1)!")
    if log_prob_squeezed.shape != (N,):
        raise AssertionError(f"FATAL: Raw log_prob shape is {log_prob_squeezed.shape}, expected ({N},)!")
    print("  --> VERIFIED: Raw 1D flow output shape is (N, 1)")

    # --- 1.6 beta_true Extraction (De-standardization) ---
    print("\n[1.6 beta_true Extraction & De-standardization]")
    std_settings = model.metadata["train_settings"]["data"]["standardization"]
    beta_mean = std_settings["mean"]["beta_true"]
    beta_std = std_settings["std"]["beta_true"]
    print(f"  Standardization factors: mean={beta_mean:.6f}, std={beta_std:.6f}")

    destandardized = sampler.transform_post(
        {"parameters": y_squeezed, "log_prob": log_prob_squeezed}
    )
    params_dict = destandardized["parameters"]
    print(f"  transform_post output parameter keys: {list(params_dict.keys())}")
    if list(params_dict.keys()) != ["beta_true"]:
        raise AssertionError(f"FATAL: transform_post parameters must only contain ['beta_true'], got {list(params_dict.keys())}!")

    beta_extracted = params_dict["beta_true"]
    print(f"  Extracted beta_true tensor shape: {beta_extracted.shape}")
    print(f"  Sample values range: [{beta_extracted.min():.4f}, {beta_extracted.max():.4f}]")

    # --- 1.7 Posterior Samples returned by BeyondGRSampler ---
    print("\n[1.7 BeyondGRSampler._run_sampler()]")
    sampler_out = sampler._run_sampler(num_samples=N, context=context)
    print(f"  _run_sampler output keys: {list(sampler_out.keys())}")
    if set(sampler_out.keys()) != {"beta_true", "log_prob"}:
        raise AssertionError(f"FATAL: _run_sampler keys must be {{'beta_true', 'log_prob'}}, got {set(sampler_out.keys())}!")

    for k, v in sampler_out.items():
        arr = np.asarray(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
        print(f"  {k:15s}: shape={arr.shape}, ndim={arr.ndim}")
        if arr.shape != (N,):
            raise AssertionError(f"FATAL: {k} shape is {arr.shape}, expected ({N},)!")
    print("  --> VERIFIED: BeyondGRSampler returns beta_true: (N,) and log_prob: (N,)")

    # --- 1.8 Final NumPy / DataFrame Representation ---
    print("\n[1.8 Final DataFrame Representation (sampler.run_sampler)]")
    sampler.context = context
    sampler.run_sampler(num_samples=N, batch_size=N)
    df = sampler.samples

    if not isinstance(df, pd.DataFrame):
        raise AssertionError(f"FATAL: sampler.samples is not a pandas DataFrame! Got {type(df)}")
    print(f"  DataFrame columns: {df.columns.tolist()}")
    print(f"  DataFrame shape  : {df.shape}")

    expected_cols = ["beta_true", "log_prob"]
    if df.columns.tolist() != expected_cols:
        raise AssertionError(f"FATAL: Expected columns {expected_cols}, got {df.columns.tolist()}!")

    beta_arr = df["beta_true"].to_numpy()
    logp_arr = df["log_prob"].to_numpy()
    if beta_arr.shape != (N,) or beta_arr.ndim != 1:
        raise AssertionError(f"FATAL: df['beta_true'] shape is {beta_arr.shape}, expected ({N},)!")
    if logp_arr.shape != (N,) or logp_arr.ndim != 1:
        raise AssertionError(f"FATAL: df['log_prob'] shape is {logp_arr.shape}, expected ({N},)!")

    # --- 1.9 Legacy Parameters Audit ---
    print("\n[1.9 Legacy Parameters Audit]")
    found_legacy = PROHIBITED_KEYS.intersection(set(df.columns))
    if found_legacy:
        raise AssertionError(f"FATAL: Prohibited legacy parameters found in DataFrame: {found_legacy}!")

    metadata_inf_params = model.metadata["train_settings"]["data"].get("inference_parameters", [])
    metadata_ctx_params = model.metadata["train_settings"]["data"].get("context_parameters", [])
    if metadata_inf_params != ["beta_true"]:
        raise AssertionError(f"FATAL: Model inference_parameters is {metadata_inf_params}, expected ['beta_true']!")
    if metadata_ctx_params:
        raise AssertionError(f"FATAL: Model context_parameters is {metadata_ctx_params}, expected []!")

    print("  No legacy parameters found in samples, columns, or model configuration.")
    print("  --> STAGE 1 TENSOR-SHAPE INTEGRITY & LEGACY AUDIT: ALL CHECKS PASSED [PASS]\n")
    return True


# ==============================================================================
# 2. INJECTED beta0 RECOVERY TEST
# ==============================================================================
def run_beta0_recovery_test(model, sampler, wfd, test_dataset, asd_dataset, ifo_list, ref_time, extrinsic_prior, args):
    """
    Evaluates the complete inference pipeline on held-out validation waveforms
    spanning the range beta_true in [-5, 5].
    """
    print("=" * 80)
    print("STAGE 2: INJECTED beta0 RECOVERY TEST ACROSS VALIDATION SET")
    print("=" * 80)

    num_val = args.num_val_waveforms
    num_samples = args.num_samples
    batch_size = args.batch_size
    print(f"Validation waveforms to evaluate: {num_val}")
    print(f"Posterior samples per waveform  : {num_samples} (batch size: {batch_size})")
    print(f"Noise injection mode            : {'Zero-noise' if args.zero_noise else 'Stationary Gaussian noise'}")

    # Extract true beta0 across all test set waveforms
    test_indices = test_dataset.indices
    all_true_beta0 = wfd.parameters.iloc[test_indices]["beta0_true"].values

    # Sort test set indices by beta0_true to span [-5, 5] uniformly
    sorted_order = np.argsort(all_true_beta0)
    selected_subindices = np.linspace(0, len(test_indices) - 1, num_val, dtype=int)
    chosen_test_indices = [test_indices[sorted_order[idx]] for idx in selected_subindices]

    print(f"Selected {len(chosen_test_indices)} validation waveforms spanning:")
    print(f"  beta0_true min: {all_true_beta0[sorted_order[0]]:.4f}")
    print(f"  beta0_true max: {all_true_beta0[sorted_order[-1]]:.4f}")

    proj = ProjectOntoDetectors(ifo_list, wfd.domain, ref_time)
    sample_ext = SampleExtrinsicParameters(extrinsic_prior)
    get_det = GetDetectorTimes(ifo_list, ref_time)
    sample_asd = SampleNoiseASD(asd_dataset)

    records = []
    t_start = time.time()

    header = (
        f"{'Sample ID':>10} | {'Injected beta0':>14} | {'Median beta0':>12} | "
        f"{'90% CI Lower':>12} | {'90% CI Upper':>12} | {'CI Width':>10} | "
        f"{'Bias':>10} | {'Covered':>8}"
    )
    print("\n" + header)
    print("-" * len(header))

    for sample_id, wfd_idx in enumerate(chosen_test_indices):
        # Retrieve raw waveform and parameters from wfd
        s = copy.deepcopy(wfd[wfd_idx])
        s["extrinsic_parameters"] = {}

        # Deterministic extrinsic parameters per sample_id for perfect reproducibility
        np.random.seed(args.seed + sample_id)
        torch.manual_seed(args.seed + sample_id)

        s = sample_ext(s)
        s = get_det(s)
        s = proj(s)
        s = sample_asd(s)

        true_beta = float(s["parameters"]["beta0_true"])

        # Construct context with or without noise
        waveform_dict = {}
        for ifo, wf in s["waveform"].items():
            if args.zero_noise:
                waveform_dict[ifo] = wf.copy()
            else:
                noise = (
                    (np.random.randn(len(wf)) + 1j * np.random.randn(len(wf)))
                    * s["asds"][ifo]
                    * wfd.domain.noise_std
                )
                waveform_dict[ifo] = wf + noise

        context = {"waveform": waveform_dict, "asds": s["asds"]}
        sampler.context = context

        # Execute actual inference pipeline
        sampler.run_sampler(num_samples=num_samples, batch_size=batch_size)
        beta_samples = sampler.samples["beta_true"].to_numpy()

        # Posterior summary statistics
        med = float(np.median(beta_samples))
        q05 = float(np.quantile(beta_samples, 0.05))
        q95 = float(np.quantile(beta_samples, 0.95))
        width = float(q95 - q05)
        bias = float(med - true_beta)
        covered = bool(q05 <= true_beta <= q95)

        records.append({
            "sample_id": sample_id,
            "wfd_index": wfd_idx,
            "beta_true": true_beta,
            "posterior_median": med,
            "posterior_lower": q05,
            "posterior_upper": q95,
            "posterior_width": width,
            "bias": bias,
            "covered": covered,
        })

        print(
            f"{sample_id:10d} | {true_beta:14.4f} | {med:12.4f} | "
            f"{q05:12.4f} | {q95:12.4f} | {width:10.4f} | "
            f"{bias:10.4f} | {str(covered):>8}"
        )

    t_elapsed = time.time() - t_start
    print("-" * len(header))
    print(f"Evaluated {num_val} waveforms in {t_elapsed:.1f} s ({t_elapsed / num_val:.2f} s/waveform).\n")

    return pd.DataFrame(records)


# ==============================================================================
# 3. PLOTTING
# ==============================================================================
def generate_diagnostic_plots(df, output_dir):
    """
    Generates:
      1. diagnostics/beta_recovery.png (median vs true with 90% CI error bars and y=x line)
      2. diagnostics/beta_bias.png (bias vs true with y=0 line)
    """
    print("=" * 80)
    print("STAGE 3: GENERATING RECOVERY & BIAS PLOTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    recovery_png_path = os.path.join(output_dir, "beta_recovery.png")
    bias_png_path = os.path.join(output_dir, "beta_bias.png")

    beta_true = df["beta_true"].values
    med = df["posterior_median"].values
    lower = df["posterior_lower"].values
    upper = df["posterior_upper"].values
    bias = df["bias"].values

    # Error bar lengths relative to median
    yerr_lower = np.maximum(0, med - lower)
    yerr_upper = np.maximum(0, upper - med)
    yerr = [yerr_lower, yerr_upper]

    # --- Plot 1: Recovery Plot (Recovered vs Injected) ---
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.errorbar(
        beta_true,
        med,
        yerr=yerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#aec7e8",
        elinewidth=1.2,
        capsize=3,
        alpha=0.85,
        label="Recovered Median & 90% CI",
    )

    # Reference y = x line
    grid_min, grid_max = -5.5, 5.5
    ax.plot(
        [grid_min, grid_max],
        [grid_min, grid_max],
        "k--",
        linewidth=1.5,
        label="Ideal Recovery ($y = x$)",
    )

    # Linear fit trendline
    slope, intercept, r_val, p_val, _ = stats.linregress(beta_true, med)
    x_fit = np.linspace(beta_true.min(), beta_true.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(
        x_fit,
        y_fit,
        "r-",
        linewidth=1.8,
        label=f"Fit: $\\hat{{\\beta}}_0 = {slope:.3f} \\beta_0 + {intercept:.3f}$ ($r = {r_val:.3f}$)",
    )

    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_xlabel(r"Injected $\beta_{0,\mathrm{true}}$", fontsize=12)
    ax.set_ylabel(r"Posterior Median $\hat{\beta}_0$", fontsize=12)
    ax.set_title(r"1D Beyond-GR Inference: $\beta_0$ Recovery Diagnostic", fontsize=14, pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(recovery_png_path)
    plt.close(fig)
    print(f"  [+] Saved recovery plot to: {recovery_png_path}")

    # --- Plot 2: Bias Plot (Bias vs Injected) ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.errorbar(
        beta_true,
        bias,
        yerr=yerr,
        fmt="s",
        color="#d62728",
        ecolor="#ff9896",
        elinewidth=1.2,
        capsize=3,
        alpha=0.85,
        label=r"Posterior Bias ($\hat{\beta}_0 - \beta_{0,\mathrm{true}}$)",
    )

    # Reference y = 0 line
    ax.axhline(0, color="k", linestyle="--", linewidth=1.5, label="Zero Bias ($y = 0$)")

    # Linear fit to bias
    slope_b, intercept_b, r_val_b, _, _ = stats.linregress(beta_true, bias)
    ax.plot(
        x_fit,
        slope_b * x_fit + intercept_b,
        "b-",
        linewidth=1.8,
        label=f"Bias Trend: slope = {slope_b:.3f} ($r = {r_val_b:.3f}$)",
    )

    ax.set_xlim(grid_min, grid_max)
    ax.set_xlabel(r"Injected $\beta_{0,\mathrm{true}}$", fontsize=12)
    ax.set_ylabel(r"Bias: $\hat{\beta}_0 - \beta_{0,\mathrm{true}}$", fontsize=12)
    ax.set_title(r"1D Beyond-GR Inference: $\beta_0$ Systematic Bias Diagnostic", fontsize=14, pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(bias_png_path)
    plt.close(fig)
    print(f"  [+] Saved bias plot to    : {bias_png_path}")


# ==============================================================================
# 4 & 5. QUANTITATIVE REPORT & SCIENTIFIC SANITY CHECKS
# ==============================================================================
def generate_diagnostic_report(df, output_dir, args):
    """
    Computes all quantitative metrics, performs scientific sanity checks,
    saves diagnostics/diagnostic_report.txt, and prints results.
    """
    print("\n" + "=" * 80)
    print("STAGE 4 & 5: QUANTITATIVE REPORT & SCIENTIFIC SANITY CHECKS")
    print("=" * 80)

    beta_true = df["beta_true"].values
    med = df["posterior_median"].values
    width = df["posterior_width"].values
    bias = df["bias"].values
    covered = df["covered"].values

    num_val = len(df)
    num_samples = args.num_samples
    inj_min, inj_max = float(beta_true.min()), float(beta_true.max())
    mean_med = float(np.mean(med))
    median_width = float(np.median(width))
    mean_abs_bias = float(np.mean(np.abs(bias)))
    median_abs_bias = float(np.median(np.abs(bias)))
    mean_signed_bias = float(np.mean(bias))

    pearson_r, pearson_p = stats.pearsonr(beta_true, med)
    spearman_rho, spearman_p = stats.spearmanr(beta_true, med)

    nominal_cov = 0.90
    emp_cov = float(np.mean(covered))

    # Scientific metrics
    slope, intercept, r_val, p_val, _ = stats.linregress(beta_true, med)
    var_ratio = float(np.var(med) / (np.var(beta_true) + 1e-12))
    width_std = float(np.std(width))

    # Boundary saturation: fractions near [-5, 5] bounds within 0.25
    lower_bound_sat = float(np.mean(med <= -4.75))
    upper_bound_sat = float(np.mean(med >= 4.75))

    report_lines = [
        "================================================================================",
        "BEYOND-GR 1D POSTERIOR RECOVERY & SCIENTIFIC DIAGNOSTIC REPORT",
        "================================================================================",
        f"Execution Timestamp          : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model Checkpoint             : {args.model_path}",
        f"Validation Waveforms Evaluated: {num_val}",
        f"Posterior Samples / Waveform : {num_samples}",
        f"Noise Injection Mode         : {'Zero-noise' if args.zero_noise else 'Stationary Gaussian noise'}",
        "",
        "--------------------------------------------------------------------------------",
        "1. QUANTITATIVE PERFORMANCE METRICS",
        "--------------------------------------------------------------------------------",
        f"Injected beta0 Range         : [{inj_min:+.4f}, {inj_max:+.4f}]",
        f"Mean Posterior Median        : {mean_med:+.4f}",
        f"Median 90% CI Width          : {median_width:.4f}",
        f"Mean Absolute Bias           : {mean_abs_bias:.4f}",
        f"Median Absolute Bias         : {median_abs_bias:.4f}",
        f"Mean Signed Bias             : {mean_signed_bias:+.4f}",
        f"Pearson Correlation (r)      : {pearson_r:+.4f} (p-value: {pearson_p:.4e})",
        f"Spearman Correlation (rho)   : {spearman_rho:+.4f} (p-value: {spearman_p:.4e})",
        f"Nominal Credible Interval    : {nominal_cov * 100:.1f}%",
        f"Empirical Coverage           : {emp_cov * 100:.1f}% ({int(emp_cov * num_val)}/{num_val} covered)",
        f"Coverage Difference          : {(emp_cov - nominal_cov) * 100:+.1f}%",
        "",
        "--------------------------------------------------------------------------------",
        "2. SCIENTIFIC SANITY CHECKS",
        "--------------------------------------------------------------------------------",
        f"Linear Regression Fit        : hat(beta0) = {slope:.4f} * beta0_true + {intercept:+.4f}",
        f"Variance Ratio Var(hat)/Var  : {var_ratio:.4f}",
        f"Posterior Width Std Dev      : {width_std:.4f}",
        f"Lower Bound Saturation (<-4.75): {lower_bound_sat * 100:.1f}%",
        f"Upper Bound Saturation (>+4.75): {upper_bound_sat * 100:.1f}%",
        "",
        "Findings & Scientific Interpretation:",
    ]

    # Analysis 1: Correlation & Median Response
    if pearson_r >= 0.70:
        corr_finding = (
            f"Strong positive correlation (r = {pearson_r:.3f}). Posterior medians track the "
            "injected Beyond-GR parameter well."
        )
    elif pearson_r >= 0.30:
        corr_finding = (
            f"Moderate positive correlation (r = {pearson_r:.3f}). Posterior medians show some sensitivity "
            "to injected beta0, but with substantial scatter or attenuation."
        )
    else:
        corr_finding = (
            f"Weak or near-zero correlation (r = {pearson_r:.3f}). Posterior medians DO NOT track injected beta0."
        )
    report_lines.append(f"  * Responsiveness: {corr_finding}")

    # Analysis 2: Posterior Collapse
    if var_ratio < 0.10:
        collapse_finding = (
            f"EVIDENCE OF POSTERIOR COLLAPSE (Var(hat(beta))/Var(beta) = {var_ratio:.4f} < 0.10). "
            f"The posterior medians hover near {mean_med:.3f} across all injections, indicating that the flow "
            "has converged to predicting approximately the marginal prior rather than a sharp conditional posterior."
        )
    else:
        collapse_finding = (
            f"No complete posterior collapse (Var(hat(beta))/Var(beta) = {var_ratio:.4f} >= 0.10). "
            "The posterior distribution shifts with injected waveform data."
        )
    report_lines.append(f"  * Distribution  : {collapse_finding}")

    # Analysis 3: Systematic Bias
    if abs(slope - 1.0) > 0.4:
        bias_finding = (
            f"Strong systematic slope deviation (slope = {slope:.3f} vs ideal 1.000). "
            f"As a consequence, bias is systematically anti-correlated with beta_true (slope ~ {slope - 1:.3f})."
        )
    else:
        bias_finding = (
            f"Slope is close to ideal (slope = {slope:.3f} ~ 1.000). Systematic bias is minimal."
        )
    report_lines.append(f"  * Bias Trend    : {bias_finding}")

    # Analysis 4: Width Variation
    if width_std < 0.5:
        width_finding = (
            f"Posterior widths are nearly invariant across injections (median width = {median_width:.2f}, "
            f"std = {width_std:.3f}). The flow outputs a constant broad uncertainty interval."
        )
    else:
        width_finding = (
            f"Posterior widths vary across injections (std = {width_std:.3f}), reflecting state-dependent uncertainty."
        )
    report_lines.append(f"  * Credible Width: {width_finding}")

    # Analysis 5: Coverage
    if abs(emp_cov - nominal_cov) <= 0.10:
        cov_finding = f"Empirical coverage ({emp_cov*100:.1f}%) matches the nominal 90% level closely."
    elif emp_cov > nominal_cov:
        cov_finding = (
            f"Empirical coverage ({emp_cov*100:.1f}%) exceeds nominal 90% (over-conservative), "
            "largely because posterior widths are broad."
        )
    else:
        cov_finding = (
            f"Empirical coverage ({emp_cov*100:.1f}%) is under-conservative compared to nominal 90%."
        )
    report_lines.append(f"  * Coverage Check: {cov_finding}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save to diagnostic_report.txt
    report_file = os.path.join(output_dir, "diagnostic_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text + "\n")
    print(f"\n  [+] Saved report to: {report_file}")

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "emp_cov": emp_cov,
        "nominal_cov": nominal_cov,
        "var_ratio": var_ratio,
        "mean_abs_bias": mean_abs_bias,
    }


# ==============================================================================
# 6. PASS / FAIL SUMMARY
# ==============================================================================
def print_pass_fail_summary(tensor_ok, legacy_ok, gen_ok, metrics):
    """
    Prints a concise PASS/FAIL table across the 6 required criteria.
    """
    # Criteria definitions:
    # 1. Tensor-shape integrity: shapes match (B, 128), (N, 1), (N,)
    status_tensor = "PASS" if tensor_ok else "FAIL"

    # 2. No legacy parameters: strictly no beta_proxy, beta_residual, chirp_mass, tilts
    status_legacy = "PASS" if legacy_ok else "FAIL"

    # 3. Posterior generation: valid numeric samples produced without NaN/inf
    status_gen = "PASS" if gen_ok else "FAIL"

    # 4. Injected-vs-recovered correlation: r >= 0.50 with p < 0.01
    r = metrics["pearson_r"]
    p = metrics["pearson_p"]
    status_corr = "PASS" if (r >= 0.50 and p < 0.01) else "FAIL (Uninformative / Marginal)"

    # 5. Credible-interval coverage: within +/- 15% of nominal 90%
    cov = metrics["emp_cov"]
    status_cov = "PASS" if abs(cov - 0.90) <= 0.15 else "FAIL"

    # 6. Overall beta0 recovery: requires both good correlation and reasonable coverage
    status_rec = "PASS" if (status_corr == "PASS" and status_cov == "PASS") else "FAIL (Marginal Prior Collapse)"

    print("\n" + "=" * 80)
    print("DIAGNOSTIC TEST SUMMARY (PASS / FAIL)")
    print("=" * 80)
    print(f"  1. Tensor-shape integrity             : [{status_tensor}]")
    print(f"  2. No legacy parameters               : [{status_legacy}]")
    print(f"  3. Posterior generation               : [{status_gen}]")
    print(f"  4. Injected-vs-recovered correlation  : [{status_corr}] (r = {r:+.4f})")
    print(f"  5. Credible-interval coverage         : [{status_cov}] ({cov*100:.1f}% vs 90%)")
    print(f"  6. beta0 recovery                     : [{status_rec}]")
    print("=" * 80 + "\n")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("1D BEYOND-GR INFERENCE PIPELINE STANDALONE DIAGNOSTIC")
    print("=" * 80)
    print(f"Model checkpoint : {args.model_path}")
    print(f"Output directory : {args.output_dir}")
    print(f"Device           : {args.device}")

    # Set master seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load trained model and create sampler
    print("\n[Loading Trained Posterior Model & BeyondGRSampler]")
    model = build_model_from_kwargs(
        filename=args.model_path, device=args.device, load_training_info=False
    )
    sampler = create_sampler(model=model)
    if not isinstance(sampler, BeyondGRSampler):
        raise AssertionError(f"Expected sampler to be BeyondGRSampler, got {type(sampler)}")

    # 2. Resolve dataset paths
    data_settings = model.metadata["train_settings"]["data"]
    wfd_path = args.wfd_path or data_settings["waveform_dataset_path"]
    asd_path = args.asd_path or model.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]

    print(f"Waveform Dataset : {wfd_path}")
    print(f"ASD Dataset      : {asd_path}")

    # 3. Load WaveformDataset & ASDDataset
    print("\n[Loading Datasets & Interferometer Transforms]")
    wfd = WaveformDataset(file_name=wfd_path)
    asd_dataset = ASDDataset(
        file_name=asd_path,
        ifos=data_settings["detectors"],
        precision="single",
        domain_update=wfd.domain.domain_dict,
    )

    extrinsic_prior = get_extrinsic_prior_dict(data_settings["extrinsic_prior"])
    ifo_list = InterferometerList(data_settings["detectors"])
    ref_time = data_settings["ref_time"]

    # 4. Split dataset into train and test sets (fixed seed 42)
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        wfd, train_fraction=data_settings.get("train_fraction", 0.9)
    )
    print(f"Dataset split: train={len(train_dataset)}, validation/test={len(test_dataset)}")

    # 5. Prepare a test sample for Stage 1 validation
    s0 = copy.deepcopy(test_dataset[0])
    s0["extrinsic_parameters"] = {}
    s0 = SampleExtrinsicParameters(extrinsic_prior)(s0)
    s0 = GetDetectorTimes(ifo_list, ref_time)(s0)
    s0 = ProjectOntoDetectors(ifo_list, wfd.domain, ref_time)(s0)
    s0 = SampleNoiseASD(asd_dataset)(s0)

    # 6. Execute Stage 1: Tensor-Shape & Legacy Audit
    tensor_ok = run_pipeline_tensor_validation(model, sampler, s0, args)
    legacy_ok = True  # Verified inside run_pipeline_tensor_validation

    # 7. Execute Stage 2: Injected beta0 Recovery Across Validation Set
    df_results = run_beta0_recovery_test(
        model=model,
        sampler=sampler,
        wfd=wfd,
        test_dataset=test_dataset,
        asd_dataset=asd_dataset,
        ifo_list=ifo_list,
        ref_time=ref_time,
        extrinsic_prior=extrinsic_prior,
        args=args,
    )

    # Save CSV output
    csv_path = os.path.join(args.output_dir, "beta_recovery.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[+] Saved recovery CSV to: {csv_path}")

    # Check posterior generation integrity
    gen_ok = bool(
        not df_results["posterior_median"].isna().any()
        and not df_results["posterior_width"].isna().any()
        and np.all(df_results["posterior_width"] > 0)
    )

    # 8. Execute Stage 3: Generate Plots
    generate_diagnostic_plots(df_results, args.output_dir)

    # 9. Execute Stage 4 & 5: Quantitative Report & Scientific Sanity Checks
    metrics = generate_diagnostic_report(df_results, args.output_dir, args)

    # 10. Print PASS / FAIL Summary
    print_pass_fail_summary(tensor_ok, legacy_ok, gen_ok, metrics)


if __name__ == "__main__":
    main()
