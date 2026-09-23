Tellinen–Madelung Magnetic Hysteresis Model
This repository contains a Python implementation of the Tellinen–Madelung magnetic hysteresis model and its application to the transient calculation of a nonlinear inductor with a ferromagnetic core.
The repository contains three related Python projects and one LTspice comparison model:
```text
Tellinen-Madelung-Model/
│
├── Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops/
├── Tellinen-Madelung-Model-Inductor/
├── LTspice_Inductor with a magnetic core/
└── Chan-Major-Loop/
```
The first project calculates symmetric and asymmetric hysteresis loops using the Tellinen model combined with Madelung return-point memory. The second project uses the same hysteresis model in the transient calculation of a nonlinear inductor. The `LTspice_Inductor with a magnetic core` folder contains the LTspice schematic used for comparison with the Python inductor calculations. The `Chan-Major-Loop` project is an auxiliary tool for generating a synthetic Chan major loop for testing and comparison.
A detailed description of the mathematical model, numerical algorithms, and implementation is given in the PDF report `Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops.pdf`.
Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops
Python project for calculating major and nested minor hysteresis loops for a prescribed magnetic-field history.
```text
Tellinen-Madelung-Model-Symmetrical&Asymmetrical-BHloops/
│
├── main.py
├── BH.py
├── Hscan.py
├── Parameters.py
├── Parameters.txt
├── Major_Loop.csv
└── BHloop.csv
```
`BH.py` contains the Tellinen–Madelung hysteresis model. `Hscan.py` defines test histories of magnetic field strength. `Major_Loop.csv` contains the tabulated magnetic characteristics of the material. `BHloop.csv` contains the calculated hysteresis trajectory.
Tellinen-Madelung-Model-Inductor
Python project for calculating transient current in an inductor with a ferromagnetic core using the Tellinen–Madelung hysteresis model.
```text
Tellinen-Madelung-Model-Inductor/
│
├── main.py
├── BH.py
├── Parameters.py
├── Parameters.txt
├── V.py
├── Major_Loop.csv
├── Current.csv
├── BHloop.csv
└── State.csv
```
`main.py` solves the coupled electrical and magnetic problem. `BH.py` contains the hysteresis model. `V.py` defines the applied voltage waveform. `Major_Loop.csv` contains the magnetic material data.
The output files are `Current.csv`, `BHloop.csv`, and `State.csv`.

LTspice_Inductor with a magnetic core
This folder contains the LTspice schematic used to simulate the inductor with a magnetic core.
```text
LTspice_Inductor with a magnetic core/
```
The LTspice simulation is used as a reference for comparison with the Python calculations of the nonlinear inductor project.
Chan-Major-Loop
Auxiliary Python project for generating a synthetic major hysteresis loop using the analytical Chan model.
```text
Chan-Major-Loop/
│
├── main.py
├── Parameters.py
├── Parameters.txt
└── Major_Loop.csv
```
The generated `Major_Loop.csv` has the same format as in the Tellinen–Madelung projects and can be used for testing and comparison. For calculations of real magnetic materials, it can be replaced by experimentally measured major-loop data.
Dependencies
```text
Python 3
NumPy
SciPy
Matplotlib
```
