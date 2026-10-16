"""Two-stage sinusoidal excitation for a major-loop demagnetization scan.

The continuous field uses cosine, so H(0) = +Hmajor on the upper
major-loop branch. Both stages end/start at the same positive field
because demagnetization is performed over an integer number of periods.
"""
import numpy as np
from Parameters import parameters as par

w = 120.0 * np.pi
T = 2.0 * np.pi / w

Hmajor = par["Hmajor"]
Hfloor = par["Hfloor"]
N_DEMAG = par["DemagPeriods"]
t_demag = N_DEMAG * T


def demag_amplitude(t):
    """Exponential fall from Hmajor towards the small positive floor."""
    t = np.asarray(t, dtype=float)
    return Hfloor + (Hmajor - Hfloor) * np.exp(-t / (3.0 * T))


# Exact envelope at the switch, used in both stages (no field jump).
Hswitch = float(demag_amplitude(t_demag))


def Hdemag(t):
    """First stage: alternating demagnetization from +Hmajor."""
    t = np.asarray(t, dtype=float)
    return demag_amplitude(t) * np.cos(w * t)


def Hscan(t, Hb):
    """Second stage, absolute time t >= t_demag: growing AC field.

    H(t_demag) = Hswitch; amplitude approaches Hb from below.
    The oscillation phase is inherited from Hdemag, without a jump.
    """
    t = np.asarray(t, dtype=float)
    tau = t - t_demag
    if np.any(tau < -1e-12):
        raise ValueError("Hscan requires t >= t_demag")
    if Hb <= Hswitch:
        raise ValueError(f"Hb ({Hb:g}) must exceed Hswitch ({Hswitch:g}) A/m")
    amplitude = Hswitch + (Hb - Hswitch) * (1.0 - np.exp(-tau / (3.0 * T)))
    return amplitude * np.cos(w * t)
