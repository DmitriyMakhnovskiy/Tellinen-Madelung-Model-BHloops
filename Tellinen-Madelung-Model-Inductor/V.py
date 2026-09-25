"""
External voltage source V(t).

All numerical voltage parameters are stored in Parameters.txt.
"""

import numpy as np
from Parameters import parameters as par


def V(t):
    return (
        par["V_offset"]
        + par["V_amplitude"]
        * np.sin(
            2.0 * np.pi * par["V_frequency"] * t
        )
    )

# PWM excitation
#
#def V(t):
#
#    V0 = par["V_test"]
#
#   if t < 0.008:
#        return +V0
#
#    elif t < 0.013:
#        return -V0
#
#    elif t < 0.016:
#        return +V0
#
#    elif t < 0.018:
#        return -V0
#
#    elif t < 0.019:
#        return +V0
#
#    elif t < 0.020:
#        return -V0
#
#    elif t < 0.022:
#        return +V0
#
#    elif t < 0.025:
#        return -V0
#
#    elif t < 0.030:
#        return +V0
#
#    else:
#        return 0.0