"""
Generation of Major_Loop.csv from the Chan hysteresis model.

The file contains:
    H_upper, B_upper
    H_lower, B_lower
    H_init,  B_init

All magnetic-model parameters are specified directly
in this file.

Dmitriy Makhnovskiy
"""

import numpy as np


# ============================================================
# CHAN MODEL PARAMETERS
# ============================================================

# Saturation magnetic flux density, T
Bs = 1.9

# Remanent magnetic flux density, T
Br = 1.0

# Coercive field strength, A/m
Hc = 25.0

# Maximum absolute field in the generated table, A/m
Hmax = 200000.0

# Number of points
N = 100001


# ============================================================
# CONSTANTS
# ============================================================

mu0 = 4.0 * np.pi * 1.0e-7


# ============================================================
# PARAMETER CHECK
# ============================================================

if Bs <= 0.0:
    raise ValueError("Bs must be positive")

if Br <= 0.0 or Br >= Bs:
    raise ValueError("Br must satisfy 0 < Br < Bs")

if Hc <= 0.0:
    raise ValueError("Hc must be positive")

if Hmax <= Hc:
    raise ValueError("Hmax should be greater than Hc")

if N < 3:
    raise ValueError("N must be at least 3")


# ============================================================
# CHAN MAJOR LOOP
# ============================================================

def Upper(H):
    """
    Upper branch of the Chan major hysteresis loop.
    """

    return (
        Bs * (H + Hc)
        /
        (
            np.abs(H + Hc)
            + Hc * (Bs / Br - 1.0)
        )
        + mu0 * H
    )


def Lower(H):
    """
    Lower branch of the Chan major hysteresis loop.
    """

    return (
        Bs * (H - Hc)
        /
        (
            np.abs(H - Hc)
            + Hc * (Bs / Br - 1.0)
        )
        + mu0 * H
    )


def Binit(H):
    """
    Initial magnetization curve of the Chan model.
    """

    return 0.5 * (Upper(H) + Lower(H))


# ============================================================
# GENERATE FIELD GRID
# ============================================================

H = np.linspace(
    -Hmax,
    Hmax,
    N
)


# ============================================================
# CALCULATE CURVES
# ============================================================

B_upper = Upper(H)
B_lower = Lower(H)
B_init = Binit(H)


# ============================================================
# CREATE OUTPUT ARRAY
# ============================================================

MajorLoop = np.column_stack(
    (
        H,
        B_upper,
        H,
        B_lower,
        H,
        B_init
    )
)


# ============================================================
# SAVE Major_Loop.csv
# ============================================================

np.savetxt(
    "Major_Loop.csv",
    MajorLoop,
    delimiter=",",
    header=(
        "H_upper,B_upper,"
        "H_lower,B_lower,"
        "H_init,B_init"
    ),
    comments="",
    fmt="%.12g"
)


print("Major_Loop.csv has been created.")
print(f"Bs   = {Bs} T")
print(f"Br   = {Br} T")
print(f"Hc   = {Hc} A/m")
print(f"Hmax = {Hmax} A/m")
print(f"N    = {N}")