from random import sample
import numpy as np
import torch
from bilby.gw.conversion import chirp_mass_and_mass_ratio_to_total_mass
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

#===========================ONLINE BEYOND-GR ROTATION===============================#
class OnlineBeyondGRRotation(object):
    """
    Applies the Beyond-GR proxy phase rotation to either:
    1) Raw unwhitened polarizations ('h_plus', 'h_cross') during dataset generation / training.
    2) Raw detector frequency-domain strains ({'H1': ..., 'L1': ...}) during inference.
    """
    def __init__(
        self,
        domain,
        total_mass=None,
        chirp_mass=None,
        pn_exponent: float = -3.0,
    ):
        self.domain = domain
        self._total_mass = total_mass
        self._chirp_mass = chirp_mass
        self.pn_exponent = pn_exponent

    @property
    def total_mass(self) -> float:
        if callable(self._total_mass):
            return float(self._total_mass())
        if self._total_mass is not None:
            return float(self._total_mass)
        if callable(self._chirp_mass):
            return float(self._chirp_mass())
        if self._chirp_mass is not None:
            return float(self._chirp_mass)
        return None

    @total_mass.setter
    def total_mass(self, value):
        self._total_mass = value

    @property
    def chirp_mass(self) -> float:
        return self.total_mass

    @chirp_mass.setter
    def chirp_mass(self, value):
        self._chirp_mass = value

    def _extract_mass(self, sample):
        """Helper to extract total_mass (or derive from chirp_mass and mass_ratio)."""
        # 1. Look for explicit total_mass
        if "extrinsic_parameters" in sample and "total_mass" in sample["extrinsic_parameters"]:
            return sample["extrinsic_parameters"]["total_mass"]
        if "parameters" in sample and "total_mass" in sample["parameters"]:
            return sample["parameters"]["total_mass"]
        if "parameters" in sample and "mass_1" in sample["parameters"] and "mass_2" in sample["parameters"]:
            return sample["parameters"]["mass_1"] + sample["parameters"]["mass_2"]
        # 2. Look for chirp_mass and mass_ratio
        p = sample.get("extrinsic_parameters", {}) or sample.get("parameters", {})
        if "chirp_mass" in p and "mass_ratio" in p:
            cm = p["chirp_mass"]
            q = p["mass_ratio"]
            return chirp_mass_and_mass_ratio_to_total_mass(cm, q)
        if "parameters" in sample and "chirp_mass" in sample["parameters"] and "mass_ratio" in sample["parameters"]:
            cm = sample["parameters"]["chirp_mass"]
            q = sample["parameters"]["mass_ratio"]
            return chirp_mass_and_mass_ratio_to_total_mass(cm, q)
        # 3. Fall back to initialized mass attribute
        if self.total_mass is not None:
            return self.total_mass
        # 4. Fall back to chirp_mass if only chirp_mass is present
        if "extrinsic_parameters" in sample and "chirp_mass" in sample["extrinsic_parameters"]:
            return sample["extrinsic_parameters"]["chirp_mass"]
        if "parameters" in sample and "chirp_mass" in sample["parameters"]:
            return sample["parameters"]["chirp_mass"]
        raise ValueError("OnlineBeyondGRRotation requires total_mass (or mass_1/mass_2, or chirp_mass/mass_ratio) in sample parameters or transform initialization.")

    def __call__(self, input_sample):
        sample = input_sample.copy()

        if "extrinsic_parameters" not in sample or "beta_proxy" not in sample["extrinsic_parameters"]:
            raise KeyError("OnlineBeyondGRRotation expects 'beta_proxy' in sample['extrinsic_parameters'].")

        beta_proxy = sample["extrinsic_parameters"]["beta_proxy"]

        # Case 1: Training mode (polarizations h_plus and h_cross present)
        if "h_plus" in sample["waveform"]:
            m_val = self._extract_mass(sample)
            freqs = self.domain.sample_frequencies
            batched, batch_size = get_batch_size_of_input_sample(input_sample)
            if batched:
                for i in range(batch_size):
                    bp_i = beta_proxy[i]
                    m_i = m_val[i] if hasattr(m_val, "__getitem__") else m_val
                    phase_factor = compute_beyond_gr_phase_factor(freqs, m_i, -bp_i, self.pn_exponent)
                    sample["waveform"]["h_plus"][i] *= phase_factor
                    sample["waveform"]["h_cross"][i] *= phase_factor
            else:
                phase_factor = compute_beyond_gr_phase_factor(freqs, m_val, -beta_proxy, self.pn_exponent)
                sample["waveform"]["h_plus"] *= phase_factor
                sample["waveform"]["h_cross"] *= phase_factor
            return sample

        # Case 2: Inference mode (detector strains by IFO)
        m_val = self._extract_mass(sample)

        print(
            f"[OnlineBeyondGRRotation] beta_proxy={beta_proxy:+.4f}  "
            f"mass={m_val:.4f} M_sun (total_mass)"
        )

        for ifo, waveform in sample["waveform"].items():
            waveform = np.asarray(waveform)
            num_bins = waveform.shape[-1]
            delta_f_event = 0.125
            frequencies = np.arange(num_bins, dtype=np.float64) * delta_f_event

            if len(frequencies) != waveform.shape[-1]:
                raise RuntimeError(
                    f"Frequency grid length {len(frequencies)} does not match waveform length {waveform.shape[-1]}."
                )

            phase_factor = compute_beyond_gr_phase_factor(
                frequency_array=frequencies,
                mass_value_solar_masses=m_val,
                coupling_parameter=-beta_proxy,
                pn_exponent=self.pn_exponent,
            )
            sample["waveform"][ifo] = waveform * phase_factor

        return sample

