import numpy as np
import lal

def compute_beyond_gr_phase_factor(
    frequency_array: np.ndarray,
    mass_value_solar_masses: float,
    coupling_parameter: float,
    pn_exponent: float,
) -> np.ndarray:
    """
    Computes the complex exponential phase factor for Beyond-GR waveforms.
    
    Formula:
    u = (pi * M_sec * f)^(1/3)
    phase_shift = coupling_parameter * u^pn_exponent
    factor = exp(i * phase_shift)

    Parameters
    ----------
    frequency_array: np.ndarray
        Frequencies at which to compute the phase shift.
    mass_value_solar_masses: float
        The mass parameter (e.g. chirp mass) in solar masses.
    coupling_parameter: float
        The coefficient of the beyond-GR phase term (e.g. beta0).
    pn_exponent: float
        The PN exponent for the term (e.g. -3 for the dipole term).

    Returns
    -------
    np.ndarray
        Complex exponential factor to multiply with the GR waveform.
    """
    if coupling_parameter == 0.0:
        return np.ones(len(frequency_array), dtype=np.complex128)

    # Convert mass to seconds
    mass_sec = mass_value_solar_masses * lal.MTSUN_SI

    # Avoid divide by zero at f=0 by handling the frequency array carefully
    # We create a mask for non-zero frequencies
    nonzero_mask = frequency_array > 0
    f_nonzero = frequency_array[nonzero_mask]

    # Calculate u
    u = (np.pi * mass_sec * f_nonzero) ** (1.0 / 3.0)
    
    # Calculate phase shift
    phase_shift = coupling_parameter * (u ** pn_exponent)

    # Calculate complex factor
    factor = np.ones(len(frequency_array), dtype=np.complex128)
    factor[nonzero_mask] = np.exp(1j * phase_shift)
    
    # At f=0, the waveform is typically 0 anyway, but let's keep the factor as 1
    # to avoid NaN * 0 = NaN issues.
    
    return factor

def inject_phase(
    polarizations: dict,
    frequency_array: np.ndarray,
    coupling_parameter: float,
    pn_exponent: float,
    mass_definition: float,
) -> dict:
    """
    Applies the beyond-GR phase shift to all polarizations in place.

    Parameters
    ----------
    polarizations: dict
        Dictionary of polarizations (e.g. {'h_plus': array, 'h_cross': array}).
        Modified in place and returned.
    frequency_array: np.ndarray
        Frequency bins.
    coupling_parameter: float
        The beyond-GR coupling parameter (beta0).
    pn_exponent: float
        The PN exponent for the term (-3).
    mass_definition: float
        The mass parameter in solar masses (chirp mass).
        
    Returns
    -------
    dict
        The modified polarizations dict.
    """
    if coupling_parameter == 0.0:
        return polarizations

    factor = compute_beyond_gr_phase_factor(
        frequency_array=frequency_array,
        mass_value_solar_masses=mass_definition,
        coupling_parameter=coupling_parameter,
        pn_exponent=pn_exponent
    )

    for pol in polarizations.keys():
        polarizations[pol] = polarizations[pol] * factor

    return polarizations

def sample_beta0() -> float:
    """
    Sample beta0_true from Uniform(-5, 5).
    """
    return np.random.uniform(-5.0, 5.0)
