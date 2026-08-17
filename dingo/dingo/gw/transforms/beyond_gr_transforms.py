from random import sample

import numpy as np
import torch
from .utils import get_batch_size_of_input_sample
from dingo.gw.beyond_gr.beyond_gr_phase import compute_beyond_gr_phase_factor

class ComputeBeyondGRParameters(object):
    """
    Computes chi1z and chi2z from a_1, a_2, tilt_1, tilt_2.
    """
    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        
        if "parameters" in sample:
            p = sample["parameters"]
            if "a_1" in p and "tilt_1" in p:
                if isinstance(p["a_1"], torch.Tensor):
                    sample["parameters"]["chi1z"] = p["a_1"] * torch.cos(p["tilt_1"])
                elif isinstance(p["a_1"], np.ndarray):
                    sample["parameters"]["chi1z"] = p["a_1"] * np.cos(p["tilt_1"])
                else:
                    sample["parameters"]["chi1z"] = p["a_1"] * np.cos(p["tilt_1"])
            
            if "a_2" in p and "tilt_2" in p:
                if isinstance(p["a_2"], torch.Tensor):
                    sample["parameters"]["chi2z"] = p["a_2"] * torch.cos(p["tilt_2"])
                elif isinstance(p["a_2"], np.ndarray):
                    sample["parameters"]["chi2z"] = p["a_2"] * np.cos(p["tilt_2"])
                else:
                    sample["parameters"]["chi2z"] = p["a_2"] * np.cos(p["tilt_2"])
        return sample


class SampleBeyondGRProxy(object):
    """
    Samples beta proxy online:
    beta_proxy = beta0_true + error (where error ~ U(-1, 1))
    beta_residual = beta_proxy - beta0_true
    """
    def __init__(self):
        pass

    def __call__(self, input_sample):
        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)
        
        if batched:
            error = np.random.uniform(-1.0, 1.0, size=batch_size).astype(np.float32)
        else:
            error = float(np.random.uniform(-1.0, 1.0))
            
        beta0_true = sample["parameters"]["beta0_true"]
        
        beta_proxy = beta0_true + error
        beta_residual = beta_proxy - beta0_true
        
        if "extrinsic_parameters" not in sample:
            sample["extrinsic_parameters"] = {}
            
        sample["extrinsic_parameters"]["beta_proxy"] = beta_proxy
        sample["extrinsic_parameters"]["beta_residual"] = beta_residual
        
        return sample

#===========================FOR TRAINING===============================#
""" class OnlineBeyondGRRotation(object):
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, input_sample):
        sample = input_sample.copy()
        
        if "h_plus" not in sample["waveform"]:
            raise ValueError("OnlineBeyondGRRotation expects polarizations 'h_plus' and 'h_cross'.")
            
        beta_proxy = sample["extrinsic_parameters"]["beta_proxy"]
        
        if "extrinsic_parameters" in sample and "chirp_mass" in sample["extrinsic_parameters"]:
            chirp_mass = sample["extrinsic_parameters"]["chirp_mass"]
        else:
            chirp_mass = sample["parameters"]["chirp_mass"]
            
        freqs = self.domain.sample_frequencies
        
        batched, batch_size = get_batch_size_of_input_sample(input_sample)
        if batched:
            for i in range(batch_size):
                bp = beta_proxy[i]
                cm = chirp_mass[i]
                # To undo injected phase, we use coupling_parameter = -beta_proxy
                phase_factor = compute_beyond_gr_phase_factor(freqs, cm, -bp, -3.0)
                
                sample["waveform"]["h_plus"][i] *= phase_factor
                sample["waveform"]["h_cross"][i] *= phase_factor
        else:
            phase_factor = compute_beyond_gr_phase_factor(freqs, chirp_mass, -beta_proxy, -3.0)
            sample["waveform"]["h_plus"] *= phase_factor
            sample["waveform"]["h_cross"] *= phase_factor

        return sample """

#===========================FOR INFERENCE===============================#
class OnlineBeyondGRRotation(object):
    """
    Applies the Beyond-GR proxy phase rotation directly to raw detector
    frequency-domain strain during inference.

    The phase factor is constructed on the SAME frequency grid as the
    incoming detector strain.

    This transform must run before whitening, repackaging, and tokenization.

    Parameters
    ----------
    domain : Domain
        Dingo domain object (used only for attribute lookup; actual freq grid
        is derived from the incoming waveform shape).
    chirp_mass : float
        Chirp mass in solar masses used for the phase rotation.  Must be set
        to the posterior median from the original Dingo-T1 run — no default.
    pn_exponent : float
        Post-Newtonian exponent (default -3.0).
    """

    def __init__(
        self,
        domain,
        chirp_mass=None,
        pn_exponent: float = -3.0,
    ):
        self.domain = domain
        self._chirp_mass = chirp_mass
        self.pn_exponent = pn_exponent

    @property
    def chirp_mass(self) -> float:
        if callable(self._chirp_mass):
            return float(self._chirp_mass())
        if self._chirp_mass is not None:
            return float(self._chirp_mass)
        raise ValueError(
            "OnlineBeyondGRRotation requires a valid chirp_mass. "
            "Pass the Dingo-T1 posterior median via BeyondGRSampler."
        )

    @chirp_mass.setter
    def chirp_mass(self, value):
        self._chirp_mass = value


    def __call__(self, input_sample):
        sample = input_sample.copy()

        # ---------------------------------------------------------
        # Retrieve beta proxy from extrinsic_parameters
        # ---------------------------------------------------------
        beta_proxy = sample["extrinsic_parameters"]["beta_proxy"]

        # Concise diagnostic — verifies the actual value reaching this transform
        print(
            f"[OnlineBeyondGRRotation] beta_proxy={beta_proxy:+.4f}  "
            f"chirp_mass={self.chirp_mass:.4f} M_sun"
        )

        # ---------------------------------------------------------
        # Rotate every detector strain
        # ---------------------------------------------------------
        for ifo, waveform in sample["waveform"].items():

            waveform = np.asarray(waveform)

            num_bins = waveform.shape[-1]
            delta_f_event = 0.125

            frequencies = (
                np.arange(num_bins, dtype=np.float64)
                * delta_f_event
            )

            if len(frequencies) != waveform.shape[-1]:
                raise RuntimeError(
                    f"Frequency grid length {len(frequencies)} does not match "
                    f"waveform length {waveform.shape[-1]}."
                )

            # -----------------------------------------------------
            # Build phase factor on the RAW EVENT frequency grid
            # -----------------------------------------------------
            phase_factor = compute_beyond_gr_phase_factor(
                frequency_array=frequencies,
                mass_value_solar_masses=self.chirp_mass,
                coupling_parameter=-beta_proxy,
                pn_exponent=self.pn_exponent,
            )

            # -----------------------------------------------------
            # Apply phase rotation
            # -----------------------------------------------------
            sample["waveform"][ifo] = waveform * phase_factor

        return sample