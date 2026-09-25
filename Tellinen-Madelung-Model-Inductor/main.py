"""
Tellinen-Madelung magnetic hysteresis model coupled to a single-loop
magnetic circuit with an optional air gap.

Unknown at each implicit time step:
    H_{n+1}

Magnetic circuit:
    N I = Lm H + Lg B / mu0

Electrical equation:
    V = R I + N A dB/dt

Backward Euler residual:
    F(Htrial) =
        R Itrial
        + N A (Btrial - Blast)/dt
        - Vnext

The crucial hysteresis rule is that every trial H is evaluated from
the SAME confirmed state at time n.  Trial states generated during
root bracketing and bisection are discarded.  Only the state associated
with the accepted root becomes the confirmed state at time n+1.

Outputs:
    Current.csv
        time_s, I_A

    BHloop.csv
        H_A_per_m, B_T

    State.csv
        time_s, root_branch, stack_depth

Dependencies:
    Python 3
    NumPy
    SciPy
    Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from BH import (
    BH_increasing,
    BH_decreasing,
    root_branch_value,
    get_H_range,
    H_INIT_MIN,
    H_INIT_MAX,
)
from Parameters import parameters as par
from V import V


# ============================================================
# CONSTANTS
# ============================================================

pi = 3.1415926535897932384626433832795
mu0 = 4.0 * pi * 1.0e-7


# ============================================================
# TIME GRID
# ============================================================

n_steps = int(round(
    par["simulation_time"] / par["dt"]
))

if not np.isclose(
    n_steps * par["dt"],
    par["simulation_time"]
):
    raise ValueError(
        "simulation_time must be an integer multiple of dt"
    )

t = (
    np.arange(n_steps + 1, dtype=float)
    * par["dt"]
)


# ============================================================
# INITIAL MAGNETIC STATE
# ============================================================

H_MIN, H_MAX = get_H_range()
Hlast = float(par["Hstart"])

if Hlast < H_MIN or Hlast > H_MAX:
    raise ValueError(
        f"Hstart={Hlast} A/m is outside the major-loop range "
        f"[{H_MIN}, {H_MAX}] A/m."
    )

root_branch = par["root_branch"]

if (
    root_branch == "initial"
    and (Hlast < H_INIT_MIN or Hlast > H_INIT_MAX)
):
    raise ValueError(
        f"Hstart={Hlast} A/m is outside the initial-curve range "
        f"[{H_INIT_MIN}, {H_INIT_MAX}] A/m."
    )

Blast = float(
    root_branch_value(
        Hlast,
        root_branch
    )
)

Ilast = (
    par["Lm"] * Hlast
    + (par["Lg"] / mu0) * Blast
) / par["N"]

reversal_stack = []

if root_branch == "upper_major":
    previous_direction = "dec"

elif root_branch == "lower_major":
    previous_direction = "inc"

else:
    if Hlast > par["H_tol"]:
        previous_direction = "inc"
    elif Hlast < -par["H_tol"]:
        previous_direction = "dec"
    else:
        previous_direction = None


# ============================================================
# ARRAYS FOR RESULTS
# ============================================================

n_points = len(t)

Ivalues = np.zeros(n_points)
Hvalues = np.zeros(n_points)
Bvalues = np.zeros(n_points)

root_history = [None] * n_points
stack_depth = np.zeros(n_points, dtype=int)

Ivalues[0] = Ilast
Hvalues[0] = Hlast
Bvalues[0] = Blast
root_history[0] = root_branch
stack_depth[0] = len(reversal_stack)


# ============================================================
# TRIAL HYSTERESIS EVALUATION
# ============================================================

def evaluate_hysteresis_trial(Htrial):

    if Htrial < H_MIN - par["H_tol"] or Htrial > H_MAX + par["H_tol"]:
        raise ValueError(
            f"Trial H={Htrial} A/m is outside Major_Loop.csv range "
            f"[{H_MIN}, {H_MAX}] A/m."
        )

    if abs(Htrial - Hlast) <= par["H_tol"]:

        trial_state = (
            root_branch,
            list(reversal_stack)
        )

        return (
            Blast,
            trial_state,
            previous_direction
        )

    if Htrial > Hlast:

        if previous_direction is None:
            was_inc = True
        else:
            was_inc = (
                previous_direction == "inc"
            )

        (
            Btrial,
            root_trial,
            stack_trial
        ) = BH_increasing(
            Htrial,
            was_inc,
            Hlast,
            Blast,
            root_branch,
            reversal_stack
        )

        trial_state = (
            root_trial,
            stack_trial
        )

        return (
            Btrial,
            trial_state,
            "inc"
        )

    if previous_direction is None:
        was_dec = True
    else:
        was_dec = (
            previous_direction == "dec"
        )

    (
        Btrial,
        root_trial,
        stack_trial
    ) = BH_decreasing(
        Htrial,
        was_dec,
        Hlast,
        Blast,
        root_branch,
        reversal_stack
    )

    trial_state = (
        root_trial,
        stack_trial
    )

    return (
        Btrial,
        trial_state,
        "dec"
    )


# ============================================================
# RESIDUAL
# ============================================================

def residual(Htrial, Vnext):

    (
        Btrial,
        trial_state,
        trial_direction
    ) = evaluate_hysteresis_trial(
        Htrial
    )

    Itrial = (
        par["Lm"] * Htrial
        + (par["Lg"] / mu0) * Btrial
    ) / par["N"]

    F = (
        par["R"] * Itrial
        + par["N"] * par["A"]
        * (Btrial - Blast)
        / par["dt"]
        - Vnext
    )

    return (
        F,
        Btrial,
        Itrial,
        trial_state,
        trial_direction
    )


# ============================================================
# BRACKET SEARCH
# ============================================================

def _find_bracket_in_direction(Vnext, F0, search_direction):
    """
    Search for a sign-changing interval starting from Hlast and moving
    monotonically in one direction.

    Important:
        Every residual evaluation still starts from the same confirmed
        hysteresis state at time n.

    Strategy:
        1. Local outward search with gradually increasing step.
        2. Check sign changes between consecutive sampled points, not only
           between Hlast and the newest point.
        3. If the expanding search reaches the Major_Loop.csv boundary
           without finding a bracket, perform a denser directional scan
           over the remaining admissible interval.

    Returns:
        (Ha, Hb, Fa, Fb) or None
    """

    if search_direction not in (-1.0, 1.0):
        raise ValueError("search_direction must be -1 or +1.")

    if search_direction > 0.0:
        H_boundary = H_MAX
    else:
        H_boundary = H_MIN

    # No room to move in this direction.
    if abs(H_boundary - Hlast) <= par["H_tol"]:
        return None

    H_prev = Hlast
    F_prev = F0

    step = par["initial_H_step"]
    growth = par["bracket_growth_factor"]

    # --------------------------------------------------------
    # Stage 1: expanding local search.
    # --------------------------------------------------------
    for _ in range(int(par["max_bracket_iter"])):

        H_candidate = H_prev + search_direction * step

        if search_direction > 0.0:
            H_candidate = min(H_candidate, H_boundary)
        else:
            H_candidate = max(H_candidate, H_boundary)

        # Numerical guard against repeated evaluation of the same H.
        if abs(H_candidate - H_prev) <= par["H_tol"]:
            break

        (
            F_candidate,
            _,
            _,
            _,
            _
        ) = residual(
            H_candidate,
            Vnext
        )

        if not np.isfinite(F_candidate):
            raise RuntimeError(
                "Non-finite residual while searching for H bracket."
            )

        # Check adjacent sampled points.
        if F_prev * F_candidate <= 0.0:
            return (
                H_prev,
                H_candidate,
                F_prev,
                F_candidate
            )

        # Boundary reached without sign change.
        if abs(H_candidate - H_boundary) <= par["H_tol"]:
            break

        H_prev = H_candidate
        F_prev = F_candidate

        step *= growth

    # --------------------------------------------------------
    # Stage 2: fallback directional scan.
    #
    # This is deliberately denser than the expanding search and
    # protects against missing a local sign change when F(H) is not
    # globally monotonic over the whole admissible H interval.
    # --------------------------------------------------------
    n_scan = int(par["bracket_scan_points"])

    if n_scan < 2:
        raise ValueError("bracket_scan_points must be at least 2.")

    H_scan = np.linspace(
        Hlast,
        H_boundary,
        n_scan + 1
    )

    H_prev = H_scan[0]
    F_prev = F0

    for H_candidate in H_scan[1:]:

        if abs(H_candidate - H_prev) <= par["H_tol"]:
            continue

        (
            F_candidate,
            _,
            _,
            _,
            _
        ) = residual(
            H_candidate,
            Vnext
        )

        if not np.isfinite(F_candidate):
            raise RuntimeError(
                "Non-finite residual during fallback H scan."
            )

        if F_prev * F_candidate <= 0.0:
            return (
                H_prev,
                H_candidate,
                F_prev,
                F_candidate
            )

        H_prev = H_candidate
        F_prev = F_candidate

    return None


# ============================================================
# SOLVE ONE TIME STEP
# ============================================================

def solve_time_step(Vnext):

    (
        F0,
        B0,
        I0,
        state0,
        direction0
    ) = residual(
        Hlast,
        Vnext
    )

    if not np.isfinite(F0):
        raise RuntimeError(
            "Non-finite residual at Hlast."
        )

    if abs(F0) <= par["residual_tol"]:
        return (
            Hlast,
            B0,
            I0,
            state0,
            direction0
        )

    # --------------------------------------------------------
    # Predict the physically expected direction.
    #
    # drive = V - R I is proportional to the required dB/dt.
    # This is only a search preference, not a hard restriction.
    # --------------------------------------------------------

    drive = (
        Vnext
        - par["R"] * Ilast
    )

    if drive > 0.0:
        primary_direction = 1.0

    elif drive < 0.0:
        primary_direction = -1.0

    else:
        if previous_direction == "dec":
            primary_direction = -1.0
        else:
            primary_direction = 1.0

    # --------------------------------------------------------
    # Search first in the physically expected direction,
    # then in the opposite direction.
    # --------------------------------------------------------

    bracket = _find_bracket_in_direction(
        Vnext,
        F0,
        primary_direction
    )

    if bracket is None:
        bracket = _find_bracket_in_direction(
            Vnext,
            F0,
            -primary_direction
        )

    if bracket is None:
        # Diagnostic residuals at both accessible data boundaries.
        (
            F_left,
            _,
            _,
            _,
            _
        ) = residual(
            H_MIN,
            Vnext
        )

        (
            F_right,
            _,
            _,
            _,
            _
        ) = residual(
            H_MAX,
            Vnext
        )

        raise RuntimeError(
            "Unable to bracket H root inside the Major_Loop.csv range.\n"
            f"timestep state: Hlast={Hlast:.9g} A/m, "
            f"Blast={Blast:.9g} T, Ilast={Ilast:.9g} A\n"
            f"Vnext={Vnext:.9g} V, drive={drive:.9g} V\n"
            f"H range=[{H_MIN:.9g}, {H_MAX:.9g}] A/m\n"
            f"F(H_MIN)={F_left:.9g} V, "
            f"F(Hlast)={F0:.9g} V, "
            f"F(H_MAX)={F_right:.9g} V\n"
            "No sign-changing interval was found after both the expanding "
            "search and the fallback scan. Extend Major_Loop.csv, reduce dt, "
            "or check circuit/source parameters."
        )

    Ha, Hb, Fa, Fb = bracket

    if Ha > Hb:
        Ha, Hb = Hb, Ha
        Fa, Fb = Fb, Fa

    # Exact endpoint root, if present.
    if abs(Fa) <= par["residual_tol"]:
        return (
            Ha,
            *residual(Ha, Vnext)[1:]
        )

    if abs(Fb) <= par["residual_tol"]:
        return (
            Hb,
            *residual(Hb, Vnext)[1:]
        )

    best = None
    best_abs_F = np.inf

    # --------------------------------------------------------
    # Bisection.
    #
    # Every midpoint is evaluated from the unchanged confirmed
    # magnetic state at time n.
    # --------------------------------------------------------

    for _ in range(
        int(par["max_bisection_iter"])
    ):

        Hm = 0.5 * (Ha + Hb)

        (
            Fm,
            Bm,
            Im,
            state_m,
            direction_m
        ) = residual(
            Hm,
            Vnext
        )

        if not np.isfinite(Fm):
            raise RuntimeError(
                "Non-finite residual during bisection."
            )

        if abs(Fm) < best_abs_F:
            best_abs_F = abs(Fm)
            best = (
                Hm,
                Bm,
                Im,
                state_m,
                direction_m
            )

        if (
            abs(Fm) <= par["residual_tol"]
            or abs(Hb - Ha) <= par["H_tol"]
        ):
            return best

        if Fa * Fm <= 0.0:
            Hb = Hm
            Fb = Fm
        else:
            Ha = Hm
            Fa = Fm

    if best is None:
        raise RuntimeError(
            "Bisection solver failed unexpectedly."
        )

    raise RuntimeError(
        "Bisection did not converge within max_bisection_iter. "
        f"Best |residual| = {best_abs_F:.6e}."
    )


# ============================================================
# TIME INTEGRATION
# ============================================================

for n in range(n_points - 1):

    Vnext = V(
        t[n + 1]
    )

    (
        Hnext,
        Bnext,
        Inext,
        state_next,
        direction_next
    ) = solve_time_step(
        Vnext
    )

    (
        root_next,
        stack_next
    ) = state_next

    # Commit the trial state ONLY after the nonlinear equation
    # has been solved.
    root_branch = root_next
    reversal_stack = list(
        stack_next
    )

    if abs(Hnext - Hlast) > par["H_tol"]:
        previous_direction = direction_next

    Ivalues[n + 1] = Inext
    Hvalues[n + 1] = Hnext
    Bvalues[n + 1] = Bnext

    root_history[n + 1] = root_branch
    stack_depth[n + 1] = len(
        reversal_stack
    )

    Hlast = Hnext
    Blast = Bnext
    Ilast = Inext


# ============================================================
# SAVE RESULTS
# ============================================================

np.savetxt(
    "Current.csv",
    np.column_stack(
        (t, Ivalues)
    ),
    delimiter=",",
    header="time_s,I_A",
    comments=""
)

np.savetxt(
    "BHloop.csv",
    np.column_stack(
        (Hvalues, Bvalues)
    ),
    delimiter=",",
    header="H_A_per_m,B_T",
    comments=""
)

with open(
    "State.csv",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "time_s,root_branch,stack_depth\n"
    )

    for ti, root, depth in zip(
        t,
        root_history,
        stack_depth
    ):
        f.write(
            f"{ti:.12g},{root},{depth}\n"
        )


# ============================================================
# PLOTS
# ============================================================

plt.figure(
    figsize=(8, 5)
)
plt.plot(
    t,
    Ivalues,
    linewidth=1.5
)
plt.xlabel(
    "Time (s)"
)
plt.ylabel(
    "Current (A)"
)
plt.title(
    "Coil Current — Tellinen-Madelung"
)
plt.grid(
    True
)
plt.tight_layout()


plt.figure(
    figsize=(7, 6)
)
plt.plot(
    Hvalues,
    Bvalues,
    linewidth=1.5
)
plt.xlabel(
    "H (A/m)"
)
plt.ylabel(
    "B (T)"
)
plt.title(
    "B-H Hysteresis Trajectory — Tellinen-Madelung"
)
plt.grid(
    True
)
plt.axhline(
    0.0,
    linewidth=0.8
)
plt.axvline(
    0.0,
    linewidth=0.8
)
plt.tight_layout()

plt.show()
