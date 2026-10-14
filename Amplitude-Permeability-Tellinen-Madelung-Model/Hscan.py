import numpy as np
from Parameters import parameters as par

pi = np.pi

# Excitation frequency.
w = 120.0 * pi
T = (2.0 * pi) / w

# Initial field amplitude. Every Hb scan starts at H(0) = Hstart.
Hstart = par["Hstart"]


def Hscan(t, Hb):
    """
    Symmetric cosine excitation whose amplitude decays from Hstart to Hb.

    Parameters
    ----------
    t : float or numpy.ndarray
        Time, s.
    Hb : float
        Final field amplitude, A/m.

    Notes
    -----
    At t = 0:
        H(0) = Hstart

    For t >> 3*T:
        H(t) -> Hb * cos(w*t)
    """
    return (
        Hb
        + (Hstart - Hb) * np.exp(-t / (3.0 * T))
    ) * np.cos(w * t)
