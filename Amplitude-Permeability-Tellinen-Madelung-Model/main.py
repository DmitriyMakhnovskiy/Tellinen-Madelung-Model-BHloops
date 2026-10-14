"""
Tellinen-Madelung amplitude-permeability calculation.

For each target field amplitude Hb, the program:
    1. starts from Hstart on the upper major-loop branch;
    2. applies Hscan(t, Hb), whose amplitude decays from Hstart to Hb;
    3. calculates the complete B(H) history with the Tellinen-Madelung model;
    4. takes only the last cosine period;
    5. finds the maximum B in that last period;
    6. calculates the amplitude permeability

           mu_a = Bmax / (mu0 * Hb);

    7. repeats the calculation for Hb decreasing from Hstart to Hstop;
    8. plots mu_a as a function of Bmax;
    9. saves the calculated data and an XY chart to
       Amplitude_Permeability.xlsx.

Dependencies:
    Python 3
    NumPy
    SciPy
    Matplotlib
    openpyxl
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from openpyxl import Workbook
from openpyxl.chart import ScatterChart, Reference
from openpyxl.chart.series_factory import SeriesFactory as Series
from openpyxl.styles import Font, Alignment

from Parameters import parameters as par
from BH import (
    BH_increasing,
    BH_decreasing,
    root_branch_value,
    H_MIN,
    H_MAX,
)
from Hscan import Hscan, T


# ============================================================
# CONSTANTS AND PARAMETERS
# ============================================================

mu0 = 4.0 * np.pi * 1.0e-7

Hstart = par["Hstart"]
Hstop = par["Hstop"]
HbPoints = par["HbPoints"]
H_tol = par["H_tol"]

# Every independent Hb calculation starts from the same magnetic state.
ROOT_BRANCH = "upper_major"

# 60 periods correspond to t_max = 1 s for w = 120*pi rad/s.
# Since the amplitude envelope has time constant 3*T, the residual
# difference from Hb at t_max is exp(-20), i.e. practically negligible.
N_PERIODS = 60
SAMPLES_PER_PERIOD = 200

t_max = N_PERIODS * T
t = np.linspace(
    0.0,
    t_max,
    N_PERIODS * SAMPLES_PER_PERIOD + 1,
)

# Exact mask for the last complete cosine period.
last_period_mask = t >= (t_max - T - 0.5 * H_tol)


# ============================================================
# SINGLE H_b SIMULATION
# ============================================================

def calculate_B_history(Hb):
    """
    Calculate B(t) for one target field amplitude Hb.

    The magnetic state is reset before every Hb calculation:
        H = Hstart,
        root_branch = upper_major,
        reversal_stack = [].
    """
    Hvalues = np.asarray(Hscan(t, Hb), dtype=float)

    # Hscan(0, Hb) must start exactly at Hstart.
    if not np.isclose(Hvalues[0], Hstart, rtol=0.0, atol=H_tol):
        raise RuntimeError(
            f"Hscan(0, Hb={Hb:g}) = {Hvalues[0]:g} A/m, "
            f"but Hstart = {Hstart:g} A/m"
        )

    # The whole requested scan must remain inside the major-loop data range.
    Hmin_requested = float(np.min(Hvalues))
    Hmax_requested = float(np.max(Hvalues))

    if Hmin_requested < H_MIN - H_tol or Hmax_requested > H_MAX + H_tol:
        raise ValueError(
            "Hscan leaves the H range available in Major_Loop.csv.\n"
            f"Available range: [{H_MIN:g}, {H_MAX:g}] A/m\n"
            f"Requested range for Hb={Hb:g}: "
            f"[{Hmin_requested:g}, {Hmax_requested:g}] A/m"
        )

    root_branch = ROOT_BRANCH
    Bstart = root_branch_value(Hstart, root_branch)

    Hlast = float(Hstart)
    Blast = float(Bstart)

    reversal_stack = []

    # The upper major branch is traversed in the decreasing-H direction.
    previous_direction = "dec"

    Bvalues = np.zeros(len(Hvalues), dtype=float)
    Bvalues[0] = Bstart

    for i, H in enumerate(Hvalues[1:], start=1):

        if np.isclose(H, Hlast, rtol=0.0, atol=H_tol):
            B = Blast

        elif H > Hlast:
            was_inc = (previous_direction == "inc")

            (
                B,
                root_branch,
                reversal_stack,
            ) = BH_increasing(
                H,
                was_inc,
                Hlast,
                Blast,
                root_branch,
                reversal_stack,
            )

            previous_direction = "inc"

        else:
            was_dec = (previous_direction == "dec")

            (
                B,
                root_branch,
                reversal_stack,
            ) = BH_decreasing(
                H,
                was_dec,
                Hlast,
                Blast,
                root_branch,
                reversal_stack,
            )

            previous_direction = "dec"

        Bvalues[i] = B
        Hlast = float(H)
        Blast = float(B)

    return Hvalues, Bvalues


# ============================================================
# EXCEL OUTPUT
# ============================================================

def save_results_to_excel(results, filename):
    """
    Save amplitude-permeability data and an XY chart to an Excel workbook.
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Amplitude permeability"

    headers = [
        "Hb (A/m)",
        "Bmax, last period (T)",
        "mu_a",
        "Hmax, last period (A/m)",
    ]

    for col, value in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col, value=value)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for row, item in enumerate(results, start=2):
        ws.cell(row=row, column=1, value=float(item["Hb"]))
        ws.cell(row=row, column=2, value=float(item["Bmax"]))
        ws.cell(row=row, column=3, value=float(item["mu_a"]))
        ws.cell(row=row, column=4, value=float(item["Hmax_last"]))

    # Calculation metadata.
    ws["F1"] = "Parameter"
    ws["G1"] = "Value"
    ws["F1"].font = Font(bold=True)
    ws["G1"].font = Font(bold=True)

    metadata = [
        ("Hstart (A/m)", Hstart),
        ("Hstop (A/m)", Hstop),
        ("HbPoints", HbPoints),
        ("mu0 (H/m)", mu0),
        ("Periods simulated", N_PERIODS),
        ("Samples per period", SAMPLES_PER_PERIOD),
        ("t_max (s)", t_max),
    ]

    for r, (name, value) in enumerate(metadata, start=2):
        ws.cell(row=r, column=6, value=name)
        ws.cell(row=r, column=7, value=float(value) if isinstance(value, (np.floating, float)) else int(value))

    # Number formatting and widths.
    for cell in ws["A"][1:]:
        cell.number_format = "0.000000"
    for cell in ws["B"][1:]:
        cell.number_format = "0.000000"
    for cell in ws["C"][1:]:
        cell.number_format = "0.000"
    for cell in ws["D"][1:]:
        cell.number_format = "0.000000"

    ws.column_dimensions["A"].width = 15
    ws.column_dimensions["B"].width = 23
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 24
    ws.column_dimensions["F"].width = 22
    ws.column_dimensions["G"].width = 18

    ws.freeze_panes = "A2"

    # XY scatter chart: amplitude permeability as a function of Bmax.
    chart = ScatterChart()
    chart.title = "Amplitude permeability vs B"
    chart.x_axis.title = "Bmax (T)"
    chart.y_axis.title = "mu_a"
    chart.height = 12
    chart.width = 20
    chart.legend = None

    xvalues = Reference(ws, min_col=2, min_row=2, max_row=len(results) + 1)
    yvalues = Reference(ws, min_col=3, min_row=2, max_row=len(results) + 1)

    series = Series(yvalues, xvalues, title="mu_a(B)")
    series.marker.symbol = "circle"
    series.marker.size = 5
    series.graphicalProperties.line.width = 19050
    chart.series.append(series)

    ws.add_chart(chart, "F10")

    wb.save(filename)


# ============================================================
# MAIN CALCULATION
# ============================================================

def main():
    # Hb must decrease from Hstart to Hstop, including both endpoints.
    Hb_values = np.linspace(Hstart, Hstop, HbPoints)

    results = []

    for index, Hb in enumerate(Hb_values, start=1):
        Hvalues, Bvalues = calculate_B_history(float(Hb))

        H_last = Hvalues[last_period_mask]
        B_last = Bvalues[last_period_mask]

        # User-requested definition:
        # take the maximum B on the final period and divide by mu0 * Hb.
        Bmax = float(np.max(B_last))
        mu_a = Bmax / (mu0 * float(Hb))

        # Diagnostic only: with the long settling time this should be
        # essentially equal to Hb.
        Hmax_last = float(np.max(H_last))

        results.append(
            {
                "Hb": float(Hb),
                "Bmax": Bmax,
                "mu_a": float(mu_a),
                "Hmax_last": Hmax_last,
            }
        )

        print(
            f"{index:3d}/{HbPoints}: "
            f"Hb = {Hb:12.6g} A/m, "
            f"Bmax = {Bmax:12.6g} T, "
            f"mu_a = {mu_a:12.6g}"
        )

    Bmax_values = np.asarray([item["Bmax"] for item in results], dtype=float)
    mu_values = np.asarray([item["mu_a"] for item in results], dtype=float)

    # Plot with increasing B from left to right.
    order = np.argsort(Bmax_values)

    plt.figure(figsize=(7, 6))
    plt.plot(
        Bmax_values[order],
        mu_values[order],
        marker="o",
        linewidth=1.5,
    )
    plt.xlabel("Bmax (T)")
    plt.ylabel("Amplitude permeability, mu_a")
    plt.title("Amplitude Permeability vs Magnetic Flux Density")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    output_path = Path(__file__).with_name("Amplitude_Permeability.xlsx")
    save_results_to_excel(results, output_path)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
