from pathlib import Path
import csv
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

from Parameters import parameters as par


# ============================================================
# NUMERICAL PARAMETERS
# ============================================================

H_tol = par["H_tol"]
B_tol = par["B_tol"]
D_tol = par["D_tol"]

IntegrationMethod = par["IntegrationMethod"]
IntegrationIntervals = par["IntegrationIntervals"]


# ============================================================
# READ Major_Loop.csv
# ============================================================

# Expected column pairs:
#   H_upper, B_upper, H_lower, B_lower [, H_init, B_init]
#
# The number of valid rows in each pair may be different.
# Empty cells are ignored independently in each pair.

_major_loop_path = Path(__file__).with_name("Major_Loop.csv")

if not _major_loop_path.exists():
    raise FileNotFoundError(
        "Major_Loop.csv is not found. The file must contain the upper "
        "and lower major-loop branches and, optionally, the initial "
        "magnetization curve."
    )


def _read_curve_pair(rows, h_col, b_col, name, required=True):
    H = []
    B = []

    for row in rows:
        if h_col >= len(row) or b_col >= len(row):
            continue

        hs = row[h_col].strip()
        bs = row[b_col].strip()

        # A pair is valid only when both cells are present.
        if not hs and not bs:
            continue

        if not hs or not bs:
            raise ValueError(
                f"Incomplete {name} point in Major_Loop.csv: "
                f"both H and B must be present in the same row"
            )

        try:
            h = float(hs)
            b = float(bs)
        except ValueError:
            # Allow one or more header rows before numerical data.
            continue

        H.append(h)
        B.append(b)

    if not H:
        if required:
            raise ValueError(
                f"No numerical data found for {name} in Major_Loop.csv"
            )
        return None, None

    if len(H) < 2:
        raise ValueError(
            f"At least two points are required for {name}"
        )

    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)

    # Sort by increasing H. Experimental files need not already be sorted.
    order = np.argsort(H)
    H = H[order]
    B = B[order]

    # Interpolation requires unique H values.
    dH = np.diff(H)
    if np.any(dH <= 0.0):
        raise ValueError(
            f"The H values for {name} must be unique"
        )

    return H, B


with open(_major_loop_path, "r", encoding="utf-8-sig", newline="") as f:
    _rows = list(csv.reader(f))

_H_upper, _B_upper = _read_curve_pair(
    _rows, 0, 1, "upper major-loop branch", required=True
)
_H_lower, _B_lower = _read_curve_pair(
    _rows, 2, 3, "lower major-loop branch", required=True
)
_H_init, _B_init = _read_curve_pair(
    _rows, 4, 5, "initial magnetization curve", required=False
)


# ============================================================
# COMMON H RANGE
# ============================================================

H_MIN = max(float(_H_upper[0]), float(_H_lower[0]))
H_MAX = min(float(_H_upper[-1]), float(_H_lower[-1]))

if H_MIN >= H_MAX:
    raise ValueError(
        "The upper and lower major-loop branches do not overlap in H"
    )

if _H_init is not None:
    H_INIT_MIN = float(_H_init[0])
    H_INIT_MAX = float(_H_init[-1])
else:
    H_INIT_MIN = H_MIN
    H_INIT_MAX = H_MAX


# ============================================================
# SHAPE-PRESERVING CUBIC INTERPOLATION
# ============================================================

# PCHIP is used instead of an unrestricted cubic spline because
# experimental B(H) branches are normally monotone in H and PCHIP
# avoids artificial overshoots between tabulated points.

_upper_interp = PchipInterpolator(_H_upper, _B_upper, extrapolate=False)
_lower_interp = PchipInterpolator(_H_lower, _B_lower, extrapolate=False)

if _H_init is not None:
    _init_interp = PchipInterpolator(_H_init, _B_init, extrapolate=False)
else:
    _init_interp = None


# ============================================================
# MAJOR LOOP AND INITIAL MAGNETIZATION CURVE
# ============================================================


def _check_common_H(H):
    arr = np.asarray(H, dtype=float)

    if np.any(arr < H_MIN - H_tol) or np.any(arr > H_MAX + H_tol):
        raise ValueError(
            f"H is outside the common major-loop range "
            f"[{H_MIN:g}, {H_MAX:g}] A/m"
        )


def _scalar_or_array(value, original):
    if np.isscalar(original):
        return float(np.asarray(value))
    return np.asarray(value, dtype=float)


def Upper(H):
    _check_common_H(H)
    value = _upper_interp(H)
    return _scalar_or_array(value, H)


def Lower(H):
    _check_common_H(H)
    value = _lower_interp(H)
    return _scalar_or_array(value, H)


def Binit(H):
    _check_common_H(H)

    if _init_interp is None:
        value = 0.5 * (np.asarray(Upper(H)) + np.asarray(Lower(H)))
        return _scalar_or_array(value, H)

    arr = np.asarray(H, dtype=float)

    if np.any(arr < H_INIT_MIN - H_tol) or np.any(arr > H_INIT_MAX + H_tol):
        raise ValueError(
            f"H is outside the initial-magnetization range "
            f"[{H_INIT_MIN:g}, {H_INIT_MAX:g}] A/m"
        )

    value = _init_interp(H)
    return _scalar_or_array(value, H)


def loop_width(H):
    D = np.asarray(Upper(H)) - np.asarray(Lower(H))

    if np.any(D <= D_tol):
        raise ValueError(
            "Upper(H) - Lower(H) is zero or too small for the "
            "Tellinen equations"
        )

    return _scalar_or_array(D, H)


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

    raise ValueError(f"Unknown root_branch: {root_branch!r}")


# ============================================================
# NORMALIZED COORDINATE y
#
# y = (B - Lower) / (Upper - Lower)
# ============================================================


def y_from_B(H, B):
    Cl = Lower(H)
    D = loop_width(H)
    y = (B - Cl) / D

    # Small numerical excursions are tolerated, but a genuinely
    # external point is considered an error.
    y_tol = max(1e-10, B_tol / max(abs(D), D_tol))

    if y < -y_tol or y > 1.0 + y_tol:
        raise ValueError(
            f"Point (H={H:g}, B={B:g}) lies outside the major loop"
        )

    return min(1.0, max(0.0, y))


def B_from_y(H, y):
    return Lower(H) + y * loop_width(H)


# ============================================================
# NUMERICAL INTEGRAL
#
# I = integral f(x) dg(x)
#
# The program implements both formulas described in the report:
#   trapezoid : piecewise-linear, O(dH^2)
#   lagrange  : 3-point quadratic Lagrange, O(dH^4)
# ============================================================


def _integral_f_dg(a, b, f_function, g_function):
    if math.isclose(a, b, rel_tol=0.0, abs_tol=H_tol):
        return 0.0

    if a < H_MIN - H_tol or a > H_MAX + H_tol:
        raise ValueError(f"Integral start H={a:g} is outside the major loop")

    if b < H_MIN - H_tol or b > H_MAX + H_tol:
        raise ValueError(f"Integral end H={b:g} is outside the major loop")

    n = int(IntegrationIntervals)

    if IntegrationMethod == "lagrange" and n % 2 != 0:
        n += 1

    x = np.linspace(a, b, n + 1)
    f = np.asarray(f_function(x), dtype=float)
    g = np.asarray(g_function(x), dtype=float)

    if IntegrationMethod == "trapezoid":
        return float(
            np.sum(
                0.5 * (f[:-1] + f[1:]) * (g[1:] - g[:-1])
            )
        )

    if IntegrationMethod == "lagrange":
        # Composite 3-point quadratic formula on [x_i, x_{i+2}]:
        #
        # I_i = 1/6 * [
        #   (3 f_i + 4 f_{i+1} - f_{i+2}) (g_{i+1} - g_i)
        # + (-f_i + 4 f_{i+1} + 3 f_{i+2}) (g_{i+2} - g_{i+1})
        # ]
        f0 = f[0:-2:2]
        f1 = f[1:-1:2]
        f2 = f[2::2]

        g0 = g[0:-2:2]
        g1 = g[1:-1:2]
        g2 = g[2::2]

        terms = (
            (3.0 * f0 + 4.0 * f1 - f2) * (g1 - g0)
            + (-f0 + 4.0 * f1 + 3.0 * f2) * (g2 - g1)
        ) / 6.0

        return float(np.sum(terms))

    raise RuntimeError(
        f"Unsupported IntegrationMethod: {IntegrationMethod!r}"
    )


def _inverse_width(H):
    return 1.0 / np.asarray(loop_width(H), dtype=float)


def integral_upper(a, b):
    """Integral_a^b [1 / D(H)] dUpper(H)."""
    return _integral_f_dg(a, b, _inverse_width, Upper)


def integral_lower(a, b):
    """Integral_a^b [1 / D(H)] dLower(H)."""
    return _integral_f_dg(a, b, _inverse_width, Lower)


# ============================================================
# RAW TELLINEN REVERSAL BRANCH
# ============================================================


def tellinen_raw_y(H, start, direction):
    Ha, Ba = start
    ya = y_from_B(Ha, Ba)

    if direction == "inc":
        I = integral_upper(Ha, H)
        return ya * math.exp(-I)

    if direction == "dec":
        I = integral_lower(Ha, H)
        return 1.0 - (1.0 - ya) * math.exp(I)

    raise ValueError("direction must be 'inc' or 'dec'")


def tellinen_raw_branch(H, start, direction):
    y = tellinen_raw_y(H, start, direction)
    return B_from_y(H, y)


# ============================================================
# TELLINEN-MADELUNG ENDPOINT-NORMALIZED BRANCH
#
# IMPORTANT:
# The endpoint correction is performed directly in B, not in the
# normalized coordinate y.  This preserves the sign of the raw
# Tellinen differential permeability whenever the scale factor is
# positive:
#
#     B_TM(H) = Ba + A_B [B_T(H) - Ba]
#
#     A_B = (Bb - Ba) / (B_T(Hb) - Ba)
#
# therefore
#
#     dB_TM/dH = A_B dB_T/dH.
#
# The previous y-normalization could stretch y so strongly that
# dB/dH became negative on some nested branches.
# ============================================================


def tellinen_madelung_branch(H, start, target, direction):
    Ha, Ba = start
    Hb, Bb = target

    # Raw Tellinen branch starting from the current reversal point.
    BT = tellinen_raw_branch(H, start, direction)
    BTb = tellinen_raw_branch(Hb, start, direction)

    denominator = BTb - Ba

    if abs(denominator) < 1e-14:
        raise RuntimeError(
            "Unable to normalize Tellinen branch in B: raw branch has "
            "zero B progress between start and target"
        )

    scale_B = (Bb - Ba) / denominator

    # A non-positive scale would reverse (or suppress) the slope of the
    # raw Tellinen branch and is incompatible with the intended monotone
    # scalar hysteresis trajectory.
    if scale_B <= 0.0:
        raise RuntimeError(
            "Tellinen-Madelung B-normalization requires a positive "
            f"scale factor, but scale_B={scale_B:g}"
        )

    B = Ba + scale_B * (BT - Ba)

    # The endpoint-normalized branch is expected to remain inside the
    # major-loop envelope.  Allow only a tiny numerical tolerance.
    Cl = Lower(H)
    Cu = Upper(H)
    bound_tol = max(1e-10, 10.0 * B_tol)

    if B < Cl - bound_tol or B > Cu + bound_tol:
        raise RuntimeError(
            "Tellinen-Madelung B-normalization produced a point outside "
            f"the major loop at H={H:g}: Lower={Cl:g}, B={B:g}, Upper={Cu:g}"
        )

    # Suppress only tiny floating-point excursions at the envelope.
    B = min(Cu, max(Cl, B))
    return B


# ============================================================
# REVERSAL-STACK HELPERS
# ============================================================


def _crossed_target(H, target_H, direction):
    if direction == "inc":
        return H >= target_H - H_tol

    if direction == "dec":
        return H <= target_H + H_tol

    raise ValueError("direction must be 'inc' or 'dec'")


def _check_branch_geometry(start, target, direction):
    Ha, _ = start
    Hb, _ = target

    if direction == "inc" and Hb <= Ha + H_tol:
        raise RuntimeError(
            "Increasing Madelung branch has a target that is not to "
            "the right of its start point"
        )

    if direction == "dec" and Hb >= Ha - H_tol:
        raise RuntimeError(
            "Decreasing Madelung branch has a target that is not to "
            "the left of its start point"
        )


# ============================================================
# ADVANCE ALONG THE CURRENT HISTORY-DEPENDENT BRANCH
# ============================================================


def _advance(H, direction, root_branch, reversal_stack):
    # Never mutate the confirmed state in-place.
    stack = list(reversal_stack)
    root_new = root_branch

    while True:

        # ----------------------------------------------------
        # NO ACTIVE REVERSAL MEMORY: ROOT BRANCH
        # ----------------------------------------------------
        if len(stack) == 0:
            B = root_branch_value(H, root_new)
            return B, root_new, stack

        # ----------------------------------------------------
        # FIRST-ORDER REVERSAL CURVE
        # ----------------------------------------------------
        if len(stack) == 1:
            P1 = stack[0]

            # There is no previous real reversal point yet, so the
            # Madelung endpoint constraint cannot be applied.  The
            # first-order reversal curve is therefore the ORIGINAL
            # (raw) Tellinen branch, irrespective of whether P1 lies
            # on Binit, the upper major branch, or the lower major
            # branch.
            #
            # This is essential for a reversal from Binit.  For the
            # Chan test data, Binit=(Upper+Lower)/2 gives y=1/2 at
            # every H.  Constraining the first branch to the symmetric
            # point (-H1,-B1) would therefore force y=1/2 everywhere
            # and would incorrectly reproduce Binit instead of a
            # hysteresis branch.
            B = tellinen_raw_branch(H, P1, direction)

            # A raw Tellinen FORC approaches the opposite major branch.
            # If the two become numerically indistinguishable, the
            # first reversal memory is exhausted and that major branch
            # becomes the new root branch.
            if direction == "inc":
                opposite_B = Lower(H)
                opposite_root = "lower_major"
            elif direction == "dec":
                opposite_B = Upper(H)
                opposite_root = "upper_major"
            else:
                raise ValueError("direction must be 'inc' or 'dec'")

            if abs(B - opposite_B) <= B_tol:
                stack.clear()
                root_new = opposite_root
                B = opposite_B

            return B, root_new, stack

        # ----------------------------------------------------
        # SECOND- AND HIGHER-ORDER REVERSAL CURVES
        # ----------------------------------------------------
        start = stack[-1]
        target = stack[-2]

        _check_branch_geometry(start, target, direction)

        if _crossed_target(H, target[0], direction):
            # The nested loop closes exactly at the stored return point.
            # Remove the start and target reversal points (wiping out),
            # then continue the same requested H step on the restored
            # parent branch. One outer step may therefore close several
            # nested loops.
            stack.pop()
            stack.pop()
            continue

        B = tellinen_madelung_branch(
            H,
            start,
            target,
            direction,
        )

        return B, root_new, stack


# ============================================================
# BH_increasing
# CURRENT H IS INCREASING
# ============================================================


def BH_increasing(
    H,
    was_inc,
    Hlast,
    Blast,
    root_branch,
    reversal_stack,
):
    stack_new = list(reversal_stack)

    # A direction change creates a real reversal point.
    if not was_inc:
        stack_new.append((float(Hlast), float(Blast)))

    return _advance(
        H,
        "inc",
        root_branch,
        stack_new,
    )


# ============================================================
# BH_decreasing
# CURRENT H IS DECREASING
# ============================================================


def BH_decreasing(
    H,
    was_dec,
    Hlast,
    Blast,
    root_branch,
    reversal_stack,
):
    stack_new = list(reversal_stack)

    # A direction change creates a real reversal point.
    if not was_dec:
        stack_new.append((float(Hlast), float(Blast)))

    return _advance(
        H,
        "dec",
        root_branch,
        stack_new,
    )
