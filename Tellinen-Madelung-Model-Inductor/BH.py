"""
Tellinen-Madelung magnetic hysteresis model.

The major hysteresis loop is read from Major_Loop.csv.
Tellinen defines the local reversal-branch shape.
Madelung return-point memory is represented by a stack of real reversal points.

The functions in this module do not modify the supplied reversal stack in-place.
This is essential inside the nonlinear inductor solver: each trial H
must start from the same confirmed magnetic state.

Dmitriy Makhnovskiy / FINGO Research Center
Project implementation prepared for the nonlinear inductor solver.
"""

from pathlib import Path
import csv
import numpy as np
from scipy.interpolate import PchipInterpolator

from Parameters import parameters as par


# ============================================================
# LOAD MAJOR LOOP
# ============================================================

_MAJOR_LOOP_FILE = Path(__file__).with_name("Major_Loop.csv")


def _read_pair(rows, h_name, b_name, required):
    H = []
    B = []

    for row in rows:
        hs = (row.get(h_name) or "").strip()
        bs = (row.get(b_name) or "").strip()

        if hs == "" and bs == "":
            continue

        if hs == "" or bs == "":
            raise ValueError(
                f"Columns {h_name}, {b_name} contain an incomplete pair."
            )

        H.append(float(hs))
        B.append(float(bs))

    if required and len(H) < 2:
        raise ValueError(
            f"Major_Loop.csv must contain at least two points in "
            f"{h_name}, {b_name}."
        )

    if not H:
        return None, None

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    order = np.argsort(H)
    H = H[order]
    B = B[order]

    if np.any(np.diff(H) <= 0.0):
        raise ValueError(
            f"Repeated or non-increasing H values found in {h_name}."
        )

    return H, B


with open(_MAJOR_LOOP_FILE, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

required_columns = {"H_upper", "B_upper", "H_lower", "B_lower"}
if not rows:
    raise ValueError("Major_Loop.csv contains no data.")

missing = required_columns.difference(rows[0].keys())
if missing:
    raise ValueError(
        "Major_Loop.csv is missing required columns: "
        + ", ".join(sorted(missing))
    )

_H_upper, _B_upper = _read_pair(
    rows, "H_upper", "B_upper", required=True
)
_H_lower, _B_lower = _read_pair(
    rows, "H_lower", "B_lower", required=True
)

has_init_columns = (
    "H_init" in rows[0]
    and "B_init" in rows[0]
)

if has_init_columns:
    _H_init, _B_init = _read_pair(
        rows, "H_init", "B_init", required=False
    )
else:
    _H_init, _B_init = None, None


# ============================================================
# INTERPOLATION
# ============================================================

_upper_interp = PchipInterpolator(
    _H_upper, _B_upper, extrapolate=False
)

_lower_interp = PchipInterpolator(
    _H_lower, _B_lower, extrapolate=False
)

if _H_init is not None:
    _init_interp = PchipInterpolator(
        _H_init, _B_init, extrapolate=False
    )
else:
    _init_interp = None


H_MIN = max(float(_H_upper[0]), float(_H_lower[0]))
H_MAX = min(float(_H_upper[-1]), float(_H_lower[-1]))

if H_MIN >= H_MAX:
    raise ValueError(
        "Upper and lower major branches have no common H interval."
    )

if _init_interp is not None:
    H_INIT_MIN = float(_H_init[0])
    H_INIT_MAX = float(_H_init[-1])
else:
    H_INIT_MIN = H_MIN
    H_INIT_MAX = H_MAX


def _return_scalar_if_scalar(x, value):
    if np.ndim(x) == 0:
        return float(np.asarray(value))
    return np.asarray(value, dtype=float)


def _check_major_range(H):
    a = np.asarray(H, dtype=float)
    if np.any(a < H_MIN - par["H_tol"]) or np.any(a > H_MAX + par["H_tol"]):
        raise ValueError(
            f"H={H} A/m is outside the common major-loop range "
            f"[{H_MIN}, {H_MAX}] A/m."
        )


def Upper(H):
    _check_major_range(H)
    return _return_scalar_if_scalar(H, _upper_interp(H))


def Lower(H):
    _check_major_range(H)
    return _return_scalar_if_scalar(H, _lower_interp(H))


def Binit(H):
    _check_major_range(H)

    if _init_interp is not None:
        a = np.asarray(H, dtype=float)
        if np.any(a < H_INIT_MIN - par["H_tol"]) or np.any(
            a > H_INIT_MAX + par["H_tol"]
        ):
            raise ValueError(
                f"H={H} A/m is outside the initial-curve range "
                f"[{H_INIT_MIN}, {H_INIT_MAX}] A/m."
            )
        value = _init_interp(H)
    else:
        value = 0.5 * (
            np.asarray(Upper(H)) + np.asarray(Lower(H))
        )

    return _return_scalar_if_scalar(H, value)


def get_H_range():
    return H_MIN, H_MAX


# ============================================================
# BASIC TELLINEN FUNCTIONS
# ============================================================

def loop_width(H):
    D = np.asarray(Upper(H)) - np.asarray(Lower(H))

    if np.any(D <= par["D_tol"]):
        raise RuntimeError(
            "Major-loop width Cu(H)-Cl(H) is too small or non-positive."
        )

    return _return_scalar_if_scalar(H, D)


def y_from_B(H, B):
    lower = float(Lower(H))
    upper = float(Upper(H))
    D = upper - lower

    if D <= par["D_tol"]:
        raise RuntimeError(
            "Major-loop width is too small for y normalization."
        )

    y = (B - lower) / D
    y_tol = par["B_tol"] / max(D, par["D_tol"])

    if y < -y_tol or y > 1.0 + y_tol:
        raise RuntimeError(
            f"Point (H={H}, B={B}) lies outside the major loop."
        )

    return float(np.clip(y, 0.0, 1.0))


def B_from_y(H, y):
    if y < -1.0e-12 or y > 1.0 + 1.0e-12:
        raise RuntimeError(f"Normalized coordinate y={y} is invalid.")

    lower = float(Lower(H))
    D = float(loop_width(H))

    return lower + float(np.clip(y, 0.0, 1.0)) * D


# ============================================================
# NUMERICAL STIELTJES INTEGRATION: integral f(H) dg(H)
# ============================================================

def _integral_f_dg(a, b, g_function):
    if abs(b - a) <= par["H_tol"]:
        return 0.0

    N = int(par["IntegrationIntervals"])

    x = np.linspace(a, b, N + 1)
    f = 1.0 / np.asarray(loop_width(x), dtype=float)
    g = np.asarray(g_function(x), dtype=float)

    method = par["IntegrationMethod"]

    if method == "trapezoid":
        return float(
            np.sum(
                0.5 * (f[:-1] + f[1:]) * (g[1:] - g[:-1])
            )
        )

    if method == "lagrange":
        total = 0.0
        i = 0

        while i + 2 <= N:
            f0, f1, f2 = f[i], f[i + 1], f[i + 2]
            g0, g1, g2 = g[i], g[i + 1], g[i + 2]

            total += (
                (3.0 * f0 + 4.0 * f1 - f2) * (g1 - g0)
                + (-f0 + 4.0 * f1 + 3.0 * f2) * (g2 - g1)
            ) / 6.0

            i += 2

        if i < N:
            total += (
                0.5 * (f[i] + f[i + 1]) * (g[i + 1] - g[i])
            )

        return float(total)

    raise ValueError(
        f"Unknown IntegrationMethod={method!r}."
    )


def integral_upper(a, b):
    return _integral_f_dg(a, b, Upper)


def integral_lower(a, b):
    return _integral_f_dg(a, b, Lower)


# ============================================================
# RAW TELLINEN REVERSAL BRANCH
# ============================================================

def tellinen_raw_y(H, start, direction):
    Ha, Ba = start

    if direction == "inc":
        if H < Ha - par["H_tol"]:
            raise RuntimeError(
                "Increasing Tellinen branch evaluated below its start H."
            )

        ya = y_from_B(Ha, Ba)
        exponent = -integral_upper(Ha, H)
        y = ya * np.exp(exponent)

    elif direction == "dec":
        if H > Ha + par["H_tol"]:
            raise RuntimeError(
                "Decreasing Tellinen branch evaluated above its start H."
            )

        ya = y_from_B(Ha, Ba)
        exponent = integral_lower(Ha, H)
        y = 1.0 - (1.0 - ya) * np.exp(exponent)

    else:
        raise ValueError("direction must be 'inc' or 'dec'.")

    if y < -1.0e-10 or y > 1.0 + 1.0e-10:
        raise RuntimeError(
            f"Raw Tellinen solution produced y={y} outside [0,1]."
        )

    return float(np.clip(y, 0.0, 1.0))


def tellinen_raw_branch(H, start, direction):
    y = tellinen_raw_y(H, start, direction)
    return B_from_y(H, y)


# ============================================================
# TELLINEN-MADELUNG ENDPOINT-CONSTRAINED BRANCH
# ============================================================

def tellinen_madelung_branch(H, start, target, direction):
    Ha, Ba = start
    Hb, Bb = target

    if direction == "inc" and Hb <= Ha + par["H_tol"]:
        raise RuntimeError(
            "For an increasing Madelung branch target H must exceed start H."
        )

    if direction == "dec" and Hb >= Ha - par["H_tol"]:
        raise RuntimeError(
            "For a decreasing Madelung branch target H must be below start H."
        )

    BT = tellinen_raw_branch(H, start, direction)
    BTb = tellinen_raw_branch(Hb, start, direction)

    denominator = BTb - Ba

    if abs(denominator) <= par["B_tol"]:
        raise RuntimeError(
            "Unable to normalize Tellinen branch: "
            "BT(target)-B(start) is too small."
        )

    A_B = (Bb - Ba) / denominator

    if A_B <= 0.0:
        raise RuntimeError(
            f"Non-positive Tellinen-Madelung scale A_B={A_B}."
        )

    B = Ba + A_B * (BT - Ba)

    lower = float(Lower(H))
    upper = float(Upper(H))

    if B < lower - par["B_tol"] or B > upper + par["B_tol"]:
        raise RuntimeError(
            "Tellinen-Madelung branch left the major-loop envelope: "
            f"H={H}, B={B}, Lower={lower}, Upper={upper}."
        )

    return float(np.clip(B, lower, upper))


# ============================================================
# ROOT BRANCH
# ============================================================

def root_branch_value(H, root_branch):
    if root_branch == "initial":
        return Binit(H)

    if root_branch == "upper_major":
        return Upper(H)

    if root_branch == "lower_major":
        return Lower(H)

    raise ValueError(
        "root_branch must be 'initial', 'upper_major', or 'lower_major'."
    )


# ============================================================
# MADELUNG MEMORY ADVANCE
# ============================================================

def _advance(H, direction, root_branch, reversal_stack):
    """
    Evaluate one H value from a copied magnetic state.

    Returns:
        B, root_branch_new, reversal_stack_new

    The supplied reversal_stack is never modified in-place.
    """

    stack = list(reversal_stack)
    root = root_branch

    while True:

        # --------------------------------------------------------
        # No active reversal memory: follow the root branch.
        # --------------------------------------------------------
        if len(stack) == 0:
            return (
                float(root_branch_value(H, root)),
                root,
                stack
            )

        # --------------------------------------------------------
        # One reversal point: first-order raw Tellinen FORC.
        # --------------------------------------------------------
        if len(stack) == 1:
            start = stack[-1]

            B = tellinen_raw_branch(
                H,
                start,
                direction
            )

            # A first-order FORC tends toward the opposite major branch.
            if direction == "inc":
                boundary = float(Lower(H))
                gap = B - boundary

                if gap <= par["B_tol"]:
                    stack.clear()
                    root = "lower_major"
                    return boundary, root, stack

            else:
                boundary = float(Upper(H))
                gap = boundary - B

                if gap <= par["B_tol"]:
                    stack.clear()
                    root = "upper_major"
                    return boundary, root, stack

            return B, root, stack

        # --------------------------------------------------------
        # Two or more reversal points:
        # endpoint-constrained Tellinen-Madelung branch.
        # --------------------------------------------------------
        start = stack[-1]
        target = stack[-2]

        Htarget, Btarget = target

        if direction == "inc":
            reached = H >= Htarget - par["H_tol"]
        else:
            reached = H <= Htarget + par["H_tol"]

        if not reached:
            B = tellinen_madelung_branch(
                H,
                start,
                target,
                direction
            )
            return B, root, stack

        # The target has been reached or crossed.
        # Remove the just-closed loop, then continue on the restored
        # parent branch.  If H is exactly the return point, preserve
        # the stored target value exactly.
        stack.pop()
        stack.pop()

        if abs(H - Htarget) <= par["H_tol"]:
            return float(Btarget), root, stack

        # If H crossed beyond the target, the while loop immediately
        # evaluates the restored parent branch at the requested H.


# ============================================================
# EXTERNAL INTERFACE
# ============================================================

def BH_increasing(
    H,
    was_inc,
    Hlast,
    Blast,
    root_branch,
    reversal_stack
):
    stack_new = list(reversal_stack)

    if not was_inc:
        stack_new.append(
            (float(Hlast), float(Blast))
        )

    return _advance(
        H,
        "inc",
        root_branch,
        stack_new
    )


def BH_decreasing(
    H,
    was_dec,
    Hlast,
    Blast,
    root_branch,
    reversal_stack
):
    stack_new = list(reversal_stack)

    if not was_dec:
        stack_new.append(
            (float(Hlast), float(Blast))
        )

    return _advance(
        H,
        "dec",
        root_branch,
        stack_new
    )
