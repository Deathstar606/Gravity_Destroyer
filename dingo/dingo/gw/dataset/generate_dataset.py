import argparse
import copy
import textwrap
import yaml
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose
from bilby.gw.prior import BBHPriorDict
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.SVD import ApplySVD, SVDBasis
from dingo.gw.transforms import WhitenFixedASD
from dingo.gw.waveform_generator import (
    NewInterfaceWaveformGenerator,
    WaveformGenerator,
    generate_waveforms_parallel,
)
from dingo.core.utils.misc import call_func_strict_output_dim
from dingo.gw.beyond_gr.beyond_gr_phase import inject_phase

class BeyondGRWaveformGeneratorWrapper:
    """Wraps a WaveformGenerator to inject Beyond-GR phase shifts."""
    def __init__(self, base_generator):
        self.base_generator = base_generator
        
    def __getattr__(self, item):
        return getattr(self.base_generator, item)
        
    def __setattr__(self, key, value):
        if key == "base_generator":
            super().__setattr__(key, value)
        else:
            setattr(self.base_generator, key, value)
            
    def generate_hplus_hcross(self, parameters):
        # Extract beyond-GR parameter
        beta_true = parameters.pop("beta_true", parameters.pop("beta0_true", 0.0))
        
        # We need total_mass for the pn_velocity computation: (pi * M_total * f)^(1/3)
        total_mass = parameters.get("total_mass")
        if total_mass is None:
            if "mass_1" in parameters and "mass_2" in parameters:
                total_mass = float(parameters["mass_1"] + parameters["mass_2"])
            elif "chirp_mass" in parameters and "mass_ratio" in parameters:
                total_mass = float(chirp_mass_and_mass_ratio_to_total_mass(
                    parameters["chirp_mass"], parameters["mass_ratio"]
                ))
            elif "chirp_mass" in parameters:
                total_mass = float(parameters["chirp_mass"])
            else:
                raise ValueError("total_mass (or mass_1/mass_2, or chirp_mass/mass_ratio) must be present in parameters")
            
        # Temporarily remove transform to inject phase on physical waveform
        transform = self.base_generator.transform
        self.base_generator.transform = None
        
        try:
            # Generate GR waveform
            polarizations = self.base_generator.generate_hplus_hcross(parameters)
            
            # Apply phase injection
            if beta_true != 0.0 and not np.any(np.isnan(polarizations["h_plus"])):
                polarizations = inject_phase(
                    polarizations=polarizations,
                    frequency_array=self.base_generator.domain(),
                    coupling_parameter=beta_true,
                    pn_exponent=-3.0,
                    mass_definition=total_mass
                )
                
            # Re-apply transform
            if transform is not None:
                polarizations = transform(polarizations)
        finally:
            self.base_generator.transform = transform
            
        parameters["beta_true"] = beta_true
        parameters["beta0_true"] = beta_true
        return polarizations

def generate_parameters_and_polarizations(
    waveform_generator: WaveformGenerator,
    prior: BBHPriorDict,
    num_samples: int,
    num_processes: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Generate a dataset of waveforms based on parameters drawn from the prior.

    Parameters
    ----------
    waveform_generator : WaveformGenerator
    prior : Prior
    num_samples : int
    num_processes : int

    Returns
    -------
    pandas DataFrame of parameters
    dictionary of numpy arrays corresponding to waveform polarizations
    """
    print("Generating dataset of size " + str(num_samples))
    parameters = pd.DataFrame(prior.sample(num_samples))
    
    # Inject Beyond-GR parameter: 90% Beyond-GR (Uniform(-5, 5)), 10% GR null-test (beta0_true = 0.0)
    num_gr = int(round(0.10 * num_samples))
    num_bgr = num_samples - num_gr
    
    beta0_true = np.zeros(num_samples, dtype=np.float64)
    if num_bgr > 0:
        beta0_true[:num_bgr] = np.random.uniform(-5.0, 5.0, size=num_bgr)
    
    parameters["beta_true"] = beta0_true
    parameters["beta0_true"] = beta0_true


    if num_processes > 1:
        with threadpool_limits(limits=1, user_api="blas"):
            with Pool(processes=num_processes) as pool:
                polarizations = generate_waveforms_parallel(
                    waveform_generator, parameters, pool
                )
    else:
        polarizations = generate_waveforms_parallel(waveform_generator, parameters)

    # Find cases where waveform generation failed and only return data for successful ones
    wf_failed = np.any(np.isnan(polarizations["h_plus"]), axis=1)
    if wf_failed.any():
        idx_failed = np.where(wf_failed)[0]
        idx_ok = np.where(~wf_failed)[0]
        polarizations_ok = {k: v[idx_ok] for k, v in polarizations.items()}
        parameters_ok = parameters.iloc[idx_ok]
        failed_percent = 100 * len(idx_failed) / len(parameters)
        print(
            f"{len(idx_failed)} out of {len(parameters)} configuration ({failed_percent:.1f}%) failed to generate."
        )
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(parameters.iloc[idx_failed])
        print(
            f"Only returning the {len(idx_ok)} successfully generated configurations."
        )
        return parameters_ok, polarizations_ok

    return parameters, polarizations


def train_svd_basis(dataset: WaveformDataset, size: int, n_train: int):
    """
    Train (and optionally validate) an SVD basis.

    Parameters
    ----------
    dataset : WaveformDataset
        Contains waveforms to be used for building SVD.
    size : int
        Number of elements to keep for the SVD basis.
    n_train : int
        Number of training waveforms to use. Remaining are used for validation. Note
        that the actual number of training waveforms is n_train * len(polarizations),
        since there is one waveform used for each polarization.

    Returns
    -------
    SVDBasis, n_train, n_test
        Since EOB waveforms can fail to generate, provide also the number used in
        training and validation.
    """
    # Prepare data for training and validation.
    train_data = np.vstack([val[:n_train] for val in dataset.polarizations.values()])
    test_data = np.vstack([val[n_train:] for val in dataset.polarizations.values()])
    test_parameters = pd.concat(
        [
            # I would like to save the polarization, but saving the dataframe with
            # string columns causes problems. Fix this later.
            # dataset.parameters.iloc[n_train:].assign(polarization=pol)
            dataset.parameters.iloc[n_train:]
            for pol in dataset.polarizations
        ]
    )
    test_parameters.reset_index(drop=True, inplace=True)

    print("Building SVD basis.")
    basis = SVDBasis()
    basis.generate_basis(train_data, size)

    assert np.allclose(basis.V[: dataset.domain.min_idx], 0)

    # Since there is a possibility that the size of the dataset returned by
    # generate_parameters_and_polarizations is smaller than requested, we don't assume
    # that there are n_test samples. Instead, we just look at the size of the test
    # dataset.
    if test_data.size != 0:
        basis.compute_test_mismatches(
            test_data, parameters=test_parameters, verbose=True
        )

    # Return also the true number of samples. Some EOB waveforms may have failed to
    # generate, so this could be smaller than the number requested.
    n_ifos = len(dataset.polarizations)
    n_train = len(train_data) // n_ifos
    n_test = len(test_data) // n_ifos

    return basis, n_train, n_test


def generate_fixed_intrinsic_parameters(
    num_per_beta: int = 50,
    beta_grid: list = None,
    fixed_config: dict = None,
) -> pd.DataFrame:
    if beta_grid is None:
        beta_grid = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    # Fixed parameters specified in user instructions
    m1 = 60.0
    m2 = 30.0
    total_m = m1 + m2
    q = m2 / m1
    chirp_m = float((m1 * m2) ** (3.0 / 5.0) / (total_m) ** (1.0 / 5.0))

    default_fixed = {
        "mass_1": m1,
        "mass_2": m2,
        "chirp_mass": chirp_m,
        "mass_ratio": q,
        "total_mass": total_m,
        "a_1": 0.5,
        "a_2": 0.5,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "luminosity_distance": 100.0,
        "theta_jn": float(np.pi / 3.0),
        "phase": 0.0,
        "geocent_time": 0.0,
    }
    if fixed_config:
        default_fixed.update(fixed_config)

    rows = []
    for b in beta_grid:
        for seed_idx in range(num_per_beta):
            row = copy.deepcopy(default_fixed)
            row["beta_true"] = float(b)
            row["beta0_true"] = float(b)
            row["noise_seed"] = int(seed_idx)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def verify_fixed_intrinsic_parameters(
    df: pd.DataFrame, validation_file: str = None
) -> None:
    import os
    fixed_keys = [
        "mass_1",
        "mass_2",
        "total_mass",
        "chirp_mass",
        "mass_ratio",
        "a_1",
        "a_2",
        "tilt_1",
        "tilt_2",
        "phi_12",
        "phi_jl",
        "luminosity_distance",
        "theta_jn",
        "phase",
        "geocent_time",
    ]
    lines = [
        "=" * 80,
        "FIXED-INTRINSIC DATASET VALIDATION REPORT",
        "=" * 80,
        f"Total Waveforms Generated : {len(df)}",
        f"Columns Present           : {list(df.columns)}",
        "",
        "--- Fixed Parameter Consistency Checks ---",
    ]
    for k in fixed_keys:
        if k in df:
            vals = df[k].unique()
            std_val = float(df[k].std())
            lines.append(
                f"  {k:20s}: fixed_value = {vals[0]:.6f}, unique_count = {len(vals)}, std = {std_val:.2e}"
            )
            if len(vals) != 1 or std_val > 1e-12:
                raise AssertionError(f"FATAL: Parameter '{k}' varies! Unique values: {vals}")
        else:
            raise AssertionError(
                f"FATAL: Required parameter '{k}' is missing from parameters!"
            )

    # Check beta_true values and counts
    beta_counts = df["beta_true"].value_counts().sort_index()
    lines.append("\n--- Beta Distribution Checks ---")
    lines.append(f"  Unique beta_true values   : {list(beta_counts.index)}")
    lines.append(f"  Counts per beta_true      : {dict(beta_counts)}")
    expected_betas = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    if list(beta_counts.index) != expected_betas:
        raise AssertionError(
            f"FATAL: Beta grid mismatch! Expected {expected_betas}, got {list(beta_counts.index)}"
        )
    if not all(c == 50 for c in beta_counts.values):
        raise AssertionError(
            f"FATAL: Expected 50 waveforms per beta value, got {dict(beta_counts)}"
        )

    # Check noise_seed structure
    lines.append("\n--- Noise Seed Pairing Checks ---")
    for b in expected_betas:
        sub = df[df["beta_true"] == b]
        seeds = sorted(sub["noise_seed"].tolist())
        if seeds != list(range(50)):
            raise AssertionError(f"FATAL: Noise seed mismatch for beta={b}!")
    lines.append(
        "  Verified paired noise seeds: Exactly seeds [0..49] present for every beta value."
    )
    lines.append("\n--> ALL FIXED-INTRINSIC VALIDATION CHECKS PASSED [SUCCESS]")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    print("\n" + report_text + "\n")
    if validation_file:
        os.makedirs(os.path.dirname(os.path.abspath(validation_file)), exist_ok=True)
        with open(validation_file, "w") as f:
            f.write(report_text + "\n")
        print(f"Saved dataset validation report to: {validation_file}")


def generate_dataset(
    settings: Dict,
    num_processes: int,
    fixed_intrinsic: bool = False,
    validation_file: str = None,
) -> WaveformDataset:
    """
    Generate a waveform dataset.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to configure the dataset
    num_processes : int
    fixed_intrinsic : bool
        If True, generate a controlled fixed-intrinsic dataset across beta_true grid.
    validation_file : str
        Optional path to save validation report for fixed-intrinsic dataset.

    Returns
    -------
    A WaveformDataset based on the settings.
    """

    prior = (
        build_prior_with_defaults(settings["intrinsic_prior"])
        if "intrinsic_prior" in settings
        else None
    )
    domain = build_domain(settings["domain"])

    new_interface_flag = settings["waveform_generator"].get("new_interface", False)
    if new_interface_flag:
        waveform_generator = NewInterfaceWaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )
    else:
        waveform_generator = WaveformGenerator(
            domain=domain,
            **settings["waveform_generator"],
        )

    # Wrap the generator to apply Beyond-GR phase injections
    waveform_generator = BeyondGRWaveformGeneratorWrapper(waveform_generator)

    dataset_dict = {"settings": settings}

    if "compression" in settings:
        compression_transforms = []

        if "whitening" in settings["compression"]:
            compression_transforms.append(
                WhitenFixedASD(
                    domain,
                    asd_file=settings["compression"]["whitening"],
                    inverse=False,
                )
            )

        if "svd" in settings["compression"]:
            svd_settings = settings["compression"]["svd"]

            # Load an SVD basis from file, if specified.
            if "file" in svd_settings:
                svd_file = svd_settings["file"]
                try:
                    basis = SVDBasis(file_name=svd_file)
                except Exception:
                    import h5py
                    with h5py.File(svd_file, "r") as f:
                        if "svd" in f:
                            svd_dict = {k: f["svd"][k][()] for k in f["svd"]}
                            basis = SVDBasis(dictionary=svd_dict)
                        else:
                            raise

            # Otherwise, generate the basis based on simulated waveforms.
            else:
                # If using whitened waveforms, then the SVD should be based on these.
                waveform_generator.transform = Compose(compression_transforms)

                n_train = svd_settings["num_training_samples"]
                n_test = svd_settings.get("num_validation_samples", 0)

                func = partial(
                    generate_parameters_and_polarizations,
                    waveform_generator,
                    prior,
                    num_processes=num_processes,
                )
                parameters, polarizations = call_func_strict_output_dim(
                    func, n_train + n_test
                )
                svd_dataset_settings = copy.deepcopy(settings)
                svd_dataset_settings["num_samples"] = len(parameters)
                del svd_dataset_settings["compression"]["svd"]

                # We build a WaveformDataset containing the SVD-training waveforms
                # because when constructed, it will automatically zero the waveforms
                # below f_min. This is useful for EOB waveforms, which are Fourier
                # transformed from time domain, and hence are nonzero below f_min. The
                # waveforms need to be zeroed below f_min because this corresponds to
                # setting the lower bound of the likelihood integral.

                svd_dataset = WaveformDataset(
                    dictionary={
                        "parameters": parameters,
                        "polarizations": polarizations,
                        "settings": svd_dataset_settings,
                    }
                )
                basis, n_train, n_test = train_svd_basis(
                    svd_dataset, svd_settings["size"], n_train
                )

            compression_transforms.append(ApplySVD(basis))
            dataset_dict["svd"] = basis.to_dictionary()

        waveform_generator.transform = Compose(compression_transforms)

    if fixed_intrinsic or settings.get("fixed_intrinsic", False):
        print("\n" + "=" * 80)
        print("GENERATING CONTROLLED FIXED-INTRINSIC DATASET")
        print("=" * 80)
        parameters = generate_fixed_intrinsic_parameters(
            num_per_beta=settings.get("num_per_beta", 50),
            beta_grid=settings.get("beta_grid", None),
            fixed_config=settings.get("fixed_intrinsic_config", None),
        )
        if num_processes > 1:
            with threadpool_limits(limits=1, user_api="blas"):
                with Pool(processes=num_processes) as pool:
                    polarizations = generate_waveforms_parallel(
                        waveform_generator, parameters, pool
                    )
        else:
            polarizations = generate_waveforms_parallel(waveform_generator, parameters)

        verify_fixed_intrinsic_parameters(parameters, validation_file)
        dataset_dict["parameters"] = parameters
        dataset_dict["polarizations"] = polarizations
        dataset_dict[len(parameters)] = len(parameters)
        dataset = WaveformDataset(dictionary=dataset_dict)
        return dataset

    func = partial(
        generate_parameters_and_polarizations,
        waveform_generator,
        prior,
        num_processes=num_processes,
    )
    parameters, polarizations = call_func_strict_output_dim(
        func, settings["num_samples"]
    )
    dataset_dict["parameters"] = parameters
    dataset_dict["polarizations"] = polarizations

    dataset_dict[settings["num_samples"]] = len(parameters)
    dataset = WaveformDataset(dictionary=dataset_dict)
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Generate a waveform dataset based on a settings file.
        """
        ),
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing database settings",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use in pool for parallel waveform generation",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="waveform_dataset.hdf5",
        help="Name of file for storing dataset.",
    )
    parser.add_argument(
        "--fixed_intrinsic",
        action="store_true",
        default=False,
        help="Generate controlled fixed-intrinsic diagnostic dataset for Beyond-GR testing",
    )
    parser.add_argument(
        "--fixed_intrinsic_validation_file",
        type=str,
        default=None,
        help="Output path for validation report",
    )
    return parser.parse_args()


def _generate_dataset_main(
    settings_file: str,
    out_file: str,
    num_processes: int,
    fixed_intrinsic: bool = False,
    validation_file: str = None,
) -> None:
    if not Path(settings_file).is_file():
        raise FileNotFoundError(f"dataset generation, failed to find {settings_file}")
    if not Path(out_file).parent.is_dir():
        raise FileNotFoundError(
            f"dataset generation: can not create {out_file}: "
            f"{Path(out_file).parent} does not exist"
        )
    # Load settings
    with open(settings_file, "r") as f:
        settings = yaml.safe_load(f)

    dataset = generate_dataset(
        settings,
        num_processes,
        fixed_intrinsic=fixed_intrinsic,
        validation_file=validation_file,
    )
    dataset.to_file(str(out_file))


def main() -> None:
    args = parse_args()
    _generate_dataset_main(
        args.settings_file,
        args.out_file,
        args.num_processes,
        fixed_intrinsic=args.fixed_intrinsic,
        validation_file=args.fixed_intrinsic_validation_file,
    )


if __name__ == "__main__":
    main()
