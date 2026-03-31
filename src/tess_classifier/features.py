from __future__ import annotations

import numpy as np


FEATURE_COLUMNS = (
    "mean_flux",
    "std_flux",
    "amplitude",
    "min_flux",
    "max_flux",
    "median_flux",
    "slope_std",
    "dip_fraction",
    "dominant_frequency",
    "dominant_power_ratio",
)


def extract_features(time_axis: np.ndarray, flux: np.ndarray) -> dict[str, float]:
    """Compute lightweight descriptive features from a light curve."""
    time_axis = np.asarray(time_axis, dtype=float)
    flux = np.asarray(flux, dtype=float)
    centered_flux = flux - np.mean(flux)

    cadence = float(np.mean(np.diff(time_axis))) if len(time_axis) > 1 else 1.0
    spectrum = np.fft.rfft(centered_flux)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(centered_flux), d=cadence)

    if len(power) > 1:
        dominant_index = int(np.argmax(power[1:]) + 1)
        nonzero_power = float(np.sum(power[1:]))
        dominant_frequency = float(frequencies[dominant_index])
        dominant_power_ratio = float(power[dominant_index] / nonzero_power) if nonzero_power else 0.0
    else:
        dominant_frequency = 0.0
        dominant_power_ratio = 0.0

    std_flux = float(np.std(flux))
    threshold = float(np.median(flux) - std_flux)

    return {
        "mean_flux": float(np.mean(flux)),
        "std_flux": std_flux,
        "amplitude": float((np.max(flux) - np.min(flux)) / 2.0),
        "min_flux": float(np.min(flux)),
        "max_flux": float(np.max(flux)),
        "median_flux": float(np.median(flux)),
        "slope_std": float(np.std(np.diff(flux))) if len(flux) > 1 else 0.0,
        "dip_fraction": float(np.mean(flux < threshold)),
        "dominant_frequency": dominant_frequency,
        "dominant_power_ratio": dominant_power_ratio,
    }

