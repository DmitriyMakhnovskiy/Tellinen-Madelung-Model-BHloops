"""
Read and validate Parameters.txt for the Tellinen-Madelung inductor model.
"""

from pathlib import Path


parameters = {}

_STRING_PARAMETERS = {
    "root_branch",
    "IntegrationMethod",
}

file_path = Path(__file__).with_name("Parameters.txt")

with open(file_path, "r", encoding="utf-8") as f:
    for raw_line in f:

        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            raise ValueError(
                f"Invalid line in Parameters.txt: {raw_line.rstrip()}"
            )

        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()

        if name in _STRING_PARAMETERS:
            parameters[name] = value
        else:
            parameters[name] = float(value)


required = {
    "root_branch",
    "Hstart",
    "N",
    "R",
    "Lm",
    "A",
    "Lg",
    "V_offset",
    "V_amplitude",
    "V_frequency",
    "simulation_time",
    "dt",
    "H_tol",
    "B_tol",
    "D_tol",
    "residual_tol",
    "max_bracket_iter",
    "max_bisection_iter",
    "initial_H_step",
    "bracket_growth_factor",
    "bracket_scan_points",
    "IntegrationMethod",
    "IntegrationIntervals",
}

missing = required.difference(parameters)
if missing:
    raise KeyError(
        "Missing Parameters.txt entries: "
        + ", ".join(sorted(missing))
    )


if parameters["root_branch"] not in {
    "initial",
    "upper_major",
    "lower_major",
}:
    raise ValueError(
        "root_branch must be initial, upper_major, or lower_major."
    )

if parameters["IntegrationMethod"] not in {
    "trapezoid",
    "lagrange",
}:
    raise ValueError(
        "IntegrationMethod must be trapezoid or lagrange."
    )

positive_parameters = [
    "N",
    "Lm",
    "A",
    "simulation_time",
    "dt",
    "H_tol",
    "B_tol",
    "D_tol",
    "residual_tol",
    "max_bracket_iter",
    "max_bisection_iter",
    "initial_H_step",
    "bracket_growth_factor",
    "bracket_scan_points",
    "IntegrationIntervals",
]

for name in positive_parameters:
    if parameters[name] <= 0.0:
        raise ValueError(
            f"Parameter {name} must be positive."
        )

if parameters["R"] < 0.0:
    raise ValueError("R must be non-negative.")

if parameters["Lg"] < 0.0:
    raise ValueError("g must be non-negative.")

if int(parameters["IntegrationIntervals"]) < 2:
    raise ValueError(
        "IntegrationIntervals must be at least 2."
    )


if parameters["bracket_growth_factor"] <= 1.0:
    raise ValueError(
        "bracket_growth_factor must be greater than 1."
    )

if int(parameters["bracket_scan_points"]) < 2:
    raise ValueError(
        "bracket_scan_points must be at least 2."
    )


def get_parameter(name):
    if name not in parameters:
        raise KeyError(
            f"Parameter '{name}' is not found in Parameters.txt"
        )

    return parameters[name]
