# Tellinen–Madelung Magnetic Hysteresis Model in Python

This repository contains a Python implementation of a magnetic hysteresis model that combines the **Tellinen differential-scaling law** with **Madelung return-point memory**. The repository also contains a nonlinear inductor solver using the same hysteresis core and an auxiliary project for generating a synthetic Chan major loop for testing.

The repository is organized into three main folders:

```text
Tellinen-Madelung-Model-BHloops/
│
├── Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops/
├── Tellinen-Madelung-Model-Inductor/
└── Chan-Major-Loop/
```

The three projects are independent but use compatible magnetic-data formats.

---

## 1. Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops

This folder contains the basic hysteresis project. Its purpose is to calculate major and nested minor \(B(H)\) loops for an externally prescribed history of magnetic field strength \(H(t)\).

The local shape of a reversal branch is determined by the Tellinen model, while the magnetic history is stored using the Madelung return-point-memory rules.

The two parts of the model therefore have different roles:

- **Tellinen** defines the local geometry of the active hysteresis branch;
- **Madelung** defines which reversal point is the current branch origin, which previous point is its return target, and when an inner loop is erased by the wiping-out rule.

The implementation uses only real reversal points. They are stored in a stack:

```text
reversal_stack = [P1, P2, ..., Pn]
```

where each point is

```text
Pi = (Hi, Bi)
```

The current branch is selected according to the number of stored reversal points:

```text
len(stack) = 0   -> root branch
len(stack) = 1   -> first-order raw Tellinen FORC
len(stack) >= 2  -> Tellinen-Madelung endpoint-constrained branch
```

For two or more reversal points, the last point is the start of the current branch and the previous point is the Madelung return point:

```text
start  = stack[-1]
target = stack[-2]
```

The raw Tellinen branch \(B_T(H)\) is corrected so that it passes exactly through the return point:

\[
B_{TM}(H)
=
B_a
+
A_B\left[B_T(H)-B_a\right],
\]

with

\[
A_B
=
\frac{B_b-B_a}
     {B_T(H_b)-B_a}.
\]

Only positive values of \(A_B\) are accepted, so that the sign of the differential permeability of the original Tellinen branch is preserved.

When a return point is reached, the corresponding inner loop is removed from the stack and the older parent branch is restored. If several return points are crossed within one numerical step, wiping-out is processed repeatedly until the correct parent branch is reached.

### Main files

```text
main.py
BH.py
Hscan.py
Parameters.py
Parameters.txt
Major_Loop.csv
BHloop.csv
```

### `BH.py`

This file contains the magnetic model itself:

- loading and interpolation of the major loop;
- `Upper(H)`, `Lower(H)` and `Binit(H)`;
- normalized Tellinen coordinate;
- numerical evaluation of the Tellinen integrals;
- raw Tellinen reversal branches;
- Tellinen-Madelung endpoint correction;
- Madelung reversal stack;
- return-point memory and wiping-out;
- `BH_increasing()` and `BH_decreasing()`.

The input major-loop data are interpolated with `scipy.interpolate.PchipInterpolator` in order to preserve the shape of the experimental curves and avoid artificial spline overshoot.

### `main.py`

The main program generates a prescribed field history using the functions in `Hscan.py` and sequentially evaluates the hysteresis state.

The confirmed magnetic state contains:

```text
Hlast
Blast
previous_direction
root_branch
reversal_stack
```

A reversal point is added only when the physical direction of \(H\) changes.

The resulting hysteresis trajectory is saved to:

```text
BHloop.csv
```

with columns

```text
H_A_per_m,B_T
```

### `Hscan.py`

This file contains test histories of magnetic field strength. They are used to generate symmetric and asymmetric nested minor loops.

The scanning functions can be replaced or modified without changing the hysteresis core.

### `Major_Loop.csv`

The magnetic material is described by tabulated major-loop data.

Required columns:

```text
H_upper,B_upper,H_lower,B_lower
```

Optional columns:

```text
H_init,B_init
```

The upper, lower and initial curves may contain different numbers of points.

If the initial magnetization curve is not supplied, the program uses

\[
B_{\mathrm{init}}(H)
=
\frac{B_{\mathrm{upper}}(H)+B_{\mathrm{lower}}(H)}{2}.
\]

This makes it possible to replace synthetic test data by experimental magnetic measurements without changing the Tellinen-Madelung algorithm.

---

## 2. Tellinen-Madelung-Model-Inductor

This folder applies the same hysteresis model to the transient calculation of a nonlinear inductor with a ferromagnetic core and an optional air gap.

In this project the magnetic field history is **not prescribed externally**. Instead, the field at every time step is obtained by solving the coupled electrical and magnetic equations.

For a magnetic path length \(L_m\), air-gap length \(L_g\), cross-sectional area \(A\), number of turns \(N\), and coil resistance \(R\),

\[
NI
=
L_m H
+
\frac{L_g}{\mu_0}B,
\]

and

\[
V
=
RI
+
NA\frac{dB}{dt}.
\]

Using Backward Euler, the nonlinear residual for a trial field value is

\[
F(H_{\mathrm{trial}})
=
R I_{\mathrm{trial}}
+
NA
\frac{B_{\mathrm{trial}}-B_n}{\Delta t}
-
V_{n+1},
\]

where

\[
I_{\mathrm{trial}}
=
\frac{
L_m H_{\mathrm{trial}}
+
(L_g/\mu_0)B_{\mathrm{trial}}
}{N}.
\]

The unknown \(H_{n+1}\) is determined from

\[
F(H_{n+1})=0.
\]

### Main numerical principle

The hysteresis model has memory. Therefore all trial values generated during root finding must be evaluated from the **same confirmed magnetic state** at time \(t_n\).

For every trial field,

\[
(H_{\mathrm{trial}},S_n)
\rightarrow
(B_{\mathrm{trial}},S_{\mathrm{trial}}),
\]

where \(S_n\) is the confirmed Tellinen-Madelung state.

The trial state is discarded unless the corresponding value of \(H_{\mathrm{trial}}\) is accepted as the final root.

This is essential: the sequence of field values generated internally by the nonlinear root solver must not become an artificial magnetization history.

Only after the nonlinear equation has converged is the corresponding trial state committed as the physical state at time \(t_{n+1}\).

### Root bracketing

Before bisection, the program searches for an interval

\[
F(H_a)F(H_b)\le 0.
\]

The preferred search direction is estimated from

\[
V_{n+1}-RI_n.
\]

The bracket search is performed in two stages.

First, the program moves outward from the confirmed field value with an initial step

```text
initial_H_step
```

and gradually increases the step using

```text
bracket_growth_factor
```

while checking the sign of the residual between every pair of consecutive trial points.

If this search does not find a sign change, a denser fallback scan is performed over the available magnetic-field interval using

```text
bracket_scan_points
```

sampling intervals.

If the bracket is not found in the preferred direction, the same procedure is repeated in the opposite direction.

The search is restricted to the field range covered by `Major_Loop.csv`. No extrapolation of the experimental magnetic data is performed.

### Bisection

After a valid bracket has been found, the root is calculated by bisection.

At every midpoint the complete Tellinen-Madelung model is evaluated again from the same confirmed state \(S_n\).

Convergence is accepted when either

\[
|F(H)| \le \texttt{residual\_tol}
\]

or

\[
|H_b-H_a| \le \texttt{H\_tol}.
\]

After convergence the accepted values

\[
H_{n+1},\quad B_{n+1},\quad I_{n+1}
\]

and the associated magnetic state become the starting state for the next time step.

### Main files

```text
main.py
BH.py
Parameters.py
Parameters.txt
V.py
Major_Loop.csv
```

### `BH.py`

The hysteresis core is the same Tellinen-Madelung implementation used in the standalone hysteresis project.

A particularly important implementation detail is that the reversal stack is copied before every trial calculation. The confirmed stack is therefore not modified by the internal iterations of the nonlinear solver.

### `V.py`

This file defines the applied voltage waveform.

The supplied implementation uses

\[
V(t)
=
V_{\mathrm{offset}}
+
V_{\mathrm{amplitude}}
\sin(2\pi f t).
\]

The waveform can be changed independently of the magnetic model.

### `Parameters.txt`

The parameter file contains:

- initial magnetic state;
- coil and magnetic-circuit geometry;
- applied-voltage parameters;
- simulation time and time step;
- Tellinen numerical-integration parameters;
- nonlinear-solver tolerances;
- bracket-search parameters.

The magnetic-circuit lengths are denoted

```text
Lm
Lg
```

to match the notation used in the analytical derivation.

### Output files

The inductor project creates:

```text
Current.csv
BHloop.csv
State.csv
```

`Current.csv`:

```text
time_s,I_A
```

`BHloop.csv`:

```text
H_A_per_m,B_T
```

`State.csv`:

```text
time_s,root_branch,stack_depth
```

`State.csv` is useful for inspecting the evolution of the Madelung memory during the transient calculation.

The program also plots the coil current \(I(t)\) and the magnetic trajectory \(B(H)\).

---

## 3. Chan-Major-Loop

This is an auxiliary project used only to generate a synthetic test major loop.

The loop is generated from the analytical Chan expressions

\[
B_{\mathrm{upper}}(H)
=
B_s
\frac{H+H_c}
{|H+H_c|+H_c(B_s/B_r-1)}
+
\mu_0 H,
\]

\[
B_{\mathrm{lower}}(H)
=
B_s
\frac{H-H_c}
{|H-H_c|+H_c(B_s/B_r-1)}
+
\mu_0 H.
\]

For testing, an initial magnetization curve can also be generated as

\[
B_{\mathrm{init}}(H)
=
\frac{
B_{\mathrm{upper}}(H)
+
B_{\mathrm{lower}}(H)
}{2}.
\]

The output is written in the same format required by the Tellinen-Madelung projects:

```text
H_upper,B_upper,H_lower,B_lower,H_init,B_init
```

The purpose of this auxiliary project is **not** to make the Tellinen-Madelung model dependent on the Chan model. It only provides a convenient reproducible reference loop for debugging, numerical tests, and comparison between different hysteresis algorithms.

For material-specific simulations, the synthetic Chan loop should be replaced by experimental major-loop data.

---

## Dependencies

The Python projects use:

```text
Python 3
NumPy
SciPy
Matplotlib
```

Install the required packages, for example:

```bash
pip install numpy scipy matplotlib
```

---

## Typical workflow

A convenient workflow for numerical testing is:

```text
1. Prepare Major_Loop.csv
       |
       +-- from experimental data
       |
       +-- or generate a synthetic test loop with Chan-Major-Loop

2. Run
   Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops
   to inspect major/minor-loop behavior and magnetic memory

3. Use the same Major_Loop.csv in
   Tellinen-Madelung-Model-Inductor
   to calculate transient coil current and the resulting B(H) trajectory
```

Using the same magnetic data in both projects makes it possible to test the hysteresis algorithm independently before introducing it into the coupled nonlinear circuit problem.

---

## Notes on the model

The implementation in this repository combines two ideas:

\[
\text{Tellinen}
\rightarrow
\text{local hysteresis-branch geometry},
\]

\[
\text{Madelung}
\rightarrow
\text{return-point memory and wiping-out}.
\]

The combined Tellinen-Madelung formulation used here is an engineering computational model developed for this project rather than a standard published model in exactly this form.

It is rate-independent: dynamic magnetic losses, eddy-current effects, and frequency-dependent material behavior are not included unless they are introduced separately.

The quality of the calculated minor loops therefore depends strongly on the quality and representativeness of the supplied major-loop data.
