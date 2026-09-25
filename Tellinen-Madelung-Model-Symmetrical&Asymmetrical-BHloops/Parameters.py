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
    "Hc",
    "Hstart",
    "root_branch",
    "H_tol",
    "B_tol",
    "D_tol",
    "IntegrationMethod",
    "IntegrationIntervals",
)

for name in _required:
    if name not in parameters:
        raise KeyError(
            f"Required parameter '{name}' is not found in Parameters.txt"
        )


# Convert integer-like numerical parameter explicitly.
if not isinstance(parameters["IntegrationIntervals"], (int, float)):
    raise TypeError("IntegrationIntervals must be numeric")

intervals_float = float(parameters["IntegrationIntervals"])
intervals_int = int(round(intervals_float))

if abs(intervals_float - intervals_int) > 1e-12:
    raise ValueError("IntegrationIntervals must be an integer")

if intervals_int < 2:
    raise ValueError("IntegrationIntervals must be at least 2")

parameters["IntegrationIntervals"] = intervals_int


# Normalize string parameters.
parameters["root_branch"] = str(parameters["root_branch"]).strip().lower()
parameters["IntegrationMethod"] = str(
    parameters["IntegrationMethod"]
).strip().lower()

_valid_branches = {
    "initial",
    "upper_major",
    "lower_major",
}

if parameters["root_branch"] not in _valid_branches:
    raise ValueError(
        "root_branch must be one of: "
        "initial, upper_major, lower_major"
    )

_valid_integration_methods = {
    "trapezoid",
    "lagrange",
}

if parameters["IntegrationMethod"] not in _valid_integration_methods:
    raise ValueError(
        "IntegrationMethod must be either 'trapezoid' or 'lagrange'"
    )


for name in ("Hc", "Hstart", "H_tol", "B_tol", "D_tol"):
    if not isinstance(parameters[name], (int, float)):
        raise TypeError(f"Parameter '{name}' must be numeric")
    parameters[name] = float(parameters[name])


if parameters["H_tol"] <= 0.0:
    raise ValueError("H_tol must be positive")

if parameters["B_tol"] <= 0.0:
    raise ValueError("B_tol must be positive")

if parameters["D_tol"] <= 0.0:
    raise ValueError("D_tol must be positive")


# ============================================================
# VALIDATE STARTING H AGAINST Major_Loop.csv
#
# The CSV file is intentionally not required to exist while the
# project skeleton is being prepared. Once it is present, the
# starting field is checked automatically at parameter import.
# ============================================================


def _read_numeric_column_range(rows, column_index):
    values = []

    for row in rows:
        if column_index >= len(row):
            continue

        cell = row[column_index].strip()
        if not cell:
            continue

        try:
            values.append(float(cell))
        except ValueError:
            # Header or non-numeric cell.
            continue

    if not values:
        return None

    return min(values), max(values)


def _validate_start_field_if_csv_exists():
    csv_path = Path(__file__).with_name("Major_Loop.csv")

    # Major_Loop.csv will be created later. Do not prevent the
    # parameter module from being inspected before that happens.
    if not csv_path.exists():
        return

    import csv

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    # Expected column pairs:
    #   0,1 : H_upper, B_upper
    #   2,3 : H_lower, B_lower
    #   4,5 : H_init,  B_init   (optional)
    upper_range = _read_numeric_column_range(rows, 0)
    lower_range = _read_numeric_column_range(rows, 2)

    if upper_range is None or lower_range is None:
        raise ValueError(
            "Major_Loop.csv must contain numeric H columns for "
            "both upper and lower major-loop branches"
        )

    # The Tellinen equations require both major branches at the same H,
    # therefore the usable range is their common overlap.
    H_min = max(upper_range[0], lower_range[0])
    H_max = min(upper_range[1], lower_range[1])

    if H_min >= H_max:
        raise ValueError(
            "Upper and lower branches in Major_Loop.csv do not have "
            "a common H interval"
        )

    Hstart = parameters["Hstart"]
    H_tol = parameters["H_tol"]

    if Hstart < H_min - H_tol or Hstart > H_max + H_tol:
        raise ValueError(
            f"Hstart={Hstart:g} A/m is outside the common major-loop "
            f"range [{H_min:g}, {H_max:g}] A/m"
        )

    # If an explicit initial-magnetization curve is supplied and the
    # selected root branch is 'initial', its range must also contain Hstart.
    if parameters["root_branch"] == "initial":
        init_range = _read_numeric_column_range(rows, 4)

        if init_range is not None:
            if Hstart < init_range[0] - H_tol or Hstart > init_range[1] + H_tol:
                raise ValueError(
                    f"Hstart={Hstart:g} A/m is outside the explicit "
                    f"initial-magnetization range "
                    f"[{init_range[0]:g}, {init_range[1]:g}] A/m"
                )


_validate_start_field_if_csv_exists()


# ============================================================
# GET PARAMETER BY NAME
# ============================================================


def get_parameter(name):
    if name not in parameters:
        raise KeyError(
            f"Parameter '{name}' is not found in Parameters.txt"
        )

    return parameters[name]
