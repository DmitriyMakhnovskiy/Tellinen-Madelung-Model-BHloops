from pathlib import Path

# ============================================================
# READ PARAMETERS FROM FILE
# ============================================================

parameters = {}

file_path = Path(__file__).with_name("Parameters.txt")


def _parse_value(text):
    """Parse numeric values as float; leave other values as strings."""
    value = text.strip()

    # Allow optional quotes around string parameters.
    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in ('"', "'")
    ):
        return value[1:-1]

    try:
        return float(value)
    except ValueError:
        return value


with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # Skip empty lines and comments.
        if not line or line.startswith("#"):
            continue

        # Remove an optional inline comment.
        if "#" in line:
            line = line.split("#", 1)[0].strip()

        if not line:
            continue

        if "=" not in line:
            raise ValueError(
                f"Invalid line in Parameters.txt: {line!r}. "
                "Expected 'name=value'."
            )

        name, value = line.split("=", 1)
        parameters[name.strip()] = _parse_value(value)


# ============================================================
# REQUIRED PARAMETERS AND BASIC VALIDATION
# ============================================================

_required = (
    "Hmajor", "Hfloor", "DemagPeriods", "HbMin", "HbMax",
    "HbPoints", "RemagPeriods", "SamplesPerPeriod",
    "H_tol", "B_tol", "D_tol", "IntegrationMethod", "IntegrationIntervals",
)

for name in _required:
    if name not in parameters:
        raise KeyError(f"Required parameter '{name}' is not found in Parameters.txt")

for name, minimum in (("HbPoints", 2), ("DemagPeriods", 1),
                      ("RemagPeriods", 1), ("SamplesPerPeriod", 20),
                      ("IntegrationIntervals", 2)):
    value = float(parameters[name])
    if not value.is_integer() or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    parameters[name] = int(value)

for name in ("Hmajor", "Hfloor", "HbMin", "HbMax", "H_tol", "B_tol", "D_tol"):
    parameters[name] = float(parameters[name])

if not (0.0 <= parameters["Hfloor"] < parameters["Hmajor"]):
    raise ValueError("Require 0 <= Hfloor < Hmajor")
if not (0 < parameters["HbMin"] <= parameters["HbMax"]):
    raise ValueError("Require 0 < HbMin <= HbMax")
if min(parameters[k] for k in ("H_tol", "B_tol", "D_tol")) <= 0:
    raise ValueError("Numerical tolerances must be positive")

import math
switch = parameters["Hfloor"] + (parameters["Hmajor"] - parameters["Hfloor"]) * math.exp(-parameters["DemagPeriods"] / 3.0)
if parameters["HbMin"] <= switch:
    raise ValueError(f"HbMin must exceed the demagnetization transition amplitude ({switch:g} A/m)")

parameters["IntegrationMethod"] = str(parameters["IntegrationMethod"]).strip().lower()
if parameters["IntegrationMethod"] not in ("trapezoid", "lagrange"):
    raise ValueError("IntegrationMethod must be 'trapezoid' or 'lagrange'")

# Check the common experimental field range, when Major_Loop.csv is available.
def _read_numeric_column_range(rows, column_index):
    values = []
    for row in rows:
        if column_index >= len(row):
            continue
        try:
            values.append(float(row[column_index]))
        except (ValueError, TypeError):
            continue
    return (min(values), max(values)) if values else None


def _validate_scan_field_if_csv_exists():
    csv_path = Path(__file__).with_name("Major_Loop.csv")
    if not csv_path.exists():
        return
    import csv
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    a = _read_numeric_column_range(rows, 0)
    b = _read_numeric_column_range(rows, 2)
    if a is None or b is None:
        raise ValueError("Major_Loop.csv must contain both major branches")
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    maximum = max(parameters["Hmajor"], parameters["HbMax"])
    if -maximum < lo - parameters["H_tol"] or maximum > hi + parameters["H_tol"]:
        raise ValueError(
            f"Scan needs [{-maximum:g}, {maximum:g}] A/m, "
            f"but major-loop branches overlap in [{lo:g}, {hi:g}] A/m"
        )

_validate_scan_field_if_csv_exists()


# ============================================================
# GET PARAMETER BY NAME
# ============================================================

def get_parameter(name):
    if name not in parameters:
        raise KeyError(
            f"Parameter '{name}' is not found in Parameters.txt"
        )

    return parameters[name]
