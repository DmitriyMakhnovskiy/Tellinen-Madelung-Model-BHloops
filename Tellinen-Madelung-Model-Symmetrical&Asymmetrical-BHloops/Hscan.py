
import numpy as np
from Parameters import parameters as par

pi = 3.1415926535897932384626433832795

# The value of the function H(t) at time t=0 must be consistent
# with the selected simulation starting point defined in the file Parameters.txt.
def Hscan1(t):
    w = 120 * pi
    T = (2 * pi) / w
    return 600 * np.sin(w * t) * np.exp(-t / (T * 3))

# Testing asymmetrical loops
Hc = par["Hc"]

# The value of the function H(t) at time t=0 must be consistent
# with the selected simulation starting point defined in the file Parameters.txt.
def Hscan2(t):

    T = t[-1] - t[0]
    tau = t - t[0]

    # Bias field
    Hbias = 0.8 * Hc

    # Initial oscillation amplitude
    A0 = 1.5 * Hc

    # Number of nested loops
    n_cycles = 5

    # Smooth establishment of the bias field
    tau_bias = 0.01 * T

    # Decay of the oscillation amplitude
    tau_decay = 0.8 * T

    omega = 2.0 * np.pi * n_cycles / T

    H = (
        Hbias * (1.0 - np.exp(-tau / tau_bias))
        +
        A0 * np.exp(-tau / tau_decay)
        * np.sin(omega * tau)
    )

    return H