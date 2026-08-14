"""
Inspect the existing Dingo T1 sampling result and extract posterior medians.
This is a diagnostic script — it reads the already-completed inference HDF5
and computes the medians we need for the Beyond-GR pipeline.
"""
import sys
sys.path.insert(0, "/home/deathstar/dingorep/dingo")

import numpy as np
import h5py

SAMPLING_HDF5 = (
    "/home/deathstar/dingorep/dingo-T1/02_inference_with_pretrained_model/"
    "events/GW190701_203306/H1/result/"
    "GW190701_203306_data0_1246048404-56_sampling.hdf5"
)

with h5py.File(SAMPLING_HDF5, "r") as f:
    samples_raw = f["samples"][()]   # structured numpy array

# Columns present
print("Columns in sampling result:", samples_raw.dtype.names)
print(f"Number of samples: {len(samples_raw)}")
print()

# Required columns
required = ["chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2",
            "luminosity_distance", "theta_jn"]

missing = [c for c in required if c not in samples_raw.dtype.names]
if missing:
    print("MISSING columns:", missing)
else:
    print("All required columns present.")
    print()

    medians = {}
    for col in required:
        medians[col] = float(np.median(samples_raw[col]))
        print(f"  {col:25s} median = {medians[col]:.6f}")

    # Derived spin projections
    chi1z = medians["a_1"] * np.cos(medians["tilt_1"])
    chi2z = medians["a_2"] * np.cos(medians["tilt_2"])
    print()
    print(f"  {'chi1z':25s} derived = {chi1z:.6f}")
    print(f"  {'chi2z':25s} derived = {chi2z:.6f}")

    print()
    print("=== SUMMARY OF MEDIANS FOR BGR CONTEXT ===")
    print(f"  chirp_mass          = {medians['chirp_mass']:.4f}")
    print(f"  mass_ratio          = {medians['mass_ratio']:.4f}")
    print(f"  chi1z               = {chi1z:.4f}")
    print(f"  chi2z               = {chi2z:.4f}")
    print(f"  luminosity_distance = {medians['luminosity_distance']:.4f}")
    print(f"  theta_jn            = {medians['theta_jn']:.4f}")
