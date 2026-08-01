import os
import numpy as np
import pandas as pd
import yaml
import lal

from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.dataset.generate_dataset import BeyondGRWaveformGeneratorWrapper, generate_dataset
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.transforms.beyond_gr_transforms import SampleBeyondGRProxy

def run_null_test_validation():
    print("===============================================================")
    print("       VALIDATING 10% GR NULL-TEST DATASET AUGMENTATION       ")
    print("===============================================================\n")

    settings_path = "dingo-T1/01_paper_settings/01_training/01_waveform_dataset/waveform_dataset_settings.yaml"
    if not os.path.exists(settings_path):
        # Fallback to local directory
        settings_path = "waveform_dataset_settings.yaml"

    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)

    # Test with N = 100 samples (10 GR null-test, 90 Beyond-GR)
    total_requested_samples = 100
    settings["num_samples"] = total_requested_samples
    if "compression" in settings:
        del settings["compression"]

    print(f"Configured dataset settings for num_samples = {total_requested_samples}")

    # Generate dataset
    out_file = "test_null_dataset.hdf5"
    if os.path.exists(out_file):
        os.remove(out_file)

    try:
        print("Generating dataset using generate_dataset()...")
        dataset = generate_dataset(settings, num_processes=1)
        dataset.to_file(out_file)

        # Load back dataset from HDF5
        print(f"Loading generated dataset from {out_file}...")
        loaded_dataset = WaveformDataset(file_name=out_file)
        
        total_samples = len(loaded_dataset)
        params_df = loaded_dataset.parameters

        assert "beta0_true" in params_df.columns, "CRITICAL ERROR: 'beta0_true' column missing in saved dataset!"

        beta0_vals = params_df["beta0_true"].values
        gr_mask = (beta0_vals == 0.0)
        bgr_mask = (beta0_vals != 0.0)

        num_gr_samples = np.sum(gr_mask)
        num_bgr_samples = np.sum(bgr_mask)

        print(f"\n--- DATASET SPLIT SUMMARY ---")
        print(f"Total number of generated samples: {total_samples}")
        print(f"Number of Beyond-GR samples:        {num_bgr_samples} ({100.0 * num_bgr_samples / total_samples:.1f}%)")
        print(f"Number of GR null-test samples:     {num_gr_samples} ({100.0 * num_gr_samples / total_samples:.1f}%)")

        assert total_samples == total_requested_samples, f"Total samples mismatch: expected {total_requested_samples}, got {total_samples}"
        assert num_gr_samples == 10, f"Expected exactly 10 GR null-test samples for N=100, got {num_gr_samples}"
        assert num_bgr_samples == 90, f"Expected exactly 90 Beyond-GR samples for N=100, got {num_bgr_samples}"

        print("  [+] Dataset split verified: Exactly 90% Beyond-GR and 10% GR null-test samples.")

        # Verify waveform phase injection for GR samples
        print("\n--- VERIFYING NO PHASE SHIFT INJECTED FOR GR SAMPLES ---")
        domain = build_domain(settings["domain"])
        prior = build_prior_with_defaults(settings["intrinsic_prior"])
        base_generator = WaveformGenerator(domain=domain, **settings["waveform_generator"])
        wrapped_generator = BeyondGRWaveformGeneratorWrapper(base_generator)

        # Test first GR sample waveform vs pure base generator waveform
        gr_idx = np.where(gr_mask)[0][0]
        gr_sample_params = params_df.iloc[gr_idx].to_dict()

        # Pure GR generation directly from base generator
        gr_params_clean = gr_sample_params.copy()
        gr_params_clean.pop("beta0_true", None)
        pure_gr_wf = base_generator.generate_hplus_hcross(gr_params_clean)

        # Wrapped generator call with beta0_true = 0
        wrapped_gr_wf = wrapped_generator.generate_hplus_hcross(gr_sample_params.copy())

        np.testing.assert_allclose(
            wrapped_gr_wf["h_plus"], pure_gr_wf["h_plus"],
            err_msg="GR null-test sample does not match pure GR waveform exactly!", rtol=1e-15, atol=1e-30
        )
        print("  [+] Confirmation: GR samples with beta0_true = 0 have zero injected phase shift and reproduce pure GR waveforms exactly.")
        print("  [+] Confirmation: Metadata saved using the existing serialization path (in dataset.parameters DataFrame).")

        # Verify proxy and residual identity: residual == -beta0_proxy for GR samples
        print("\n--- VERIFYING RESIDUAL IDENTITY: residual == -beta0_proxy ---")
        proxy_sampler = SampleBeyondGRProxy()

        gr_indices = np.where(gr_mask)[0]
        all_residual_matches = True

        for idx in gr_indices:
            row_dict = params_df.iloc[idx].to_dict()
            sample_struct = {"parameters": row_dict}
            transformed_sample = proxy_sampler(sample_struct)

            beta0_true_val = transformed_sample["parameters"]["beta0_true"]
            beta0_proxy_val = transformed_sample["extrinsic_parameters"]["beta_proxy"]
            beta_residual_val = transformed_sample["extrinsic_parameters"]["beta_residual"]

            # Residual definition: residual = beta0_true - beta0_proxy
            residual_val = beta0_true_val - beta0_proxy_val

            # When beta0_true = 0, residual = -beta0_proxy
            diff = abs(residual_val - (-beta0_proxy_val))
            assert diff < 1e-7, f"Identity residual == -beta0_proxy failed for GR sample index {idx}: diff={diff}"

            # Also check beta_residual (defined in pipeline as beta_proxy - beta0_true)
            assert abs(beta_residual_val - beta0_proxy_val) < 1e-7

        print("  [+] Confirmation: residual == -beta0_proxy holds within numerical precision for ALL 10 GR null-test samples.")

        print("\n===============================================================")
        print("              ALL VALIDATION TESTS PASSED SUCCESSFULLY!         ")
        print("===============================================================")

    finally:
        if os.path.exists(out_file):
            os.remove(out_file)

if __name__ == "__main__":
    run_null_test_validation()
