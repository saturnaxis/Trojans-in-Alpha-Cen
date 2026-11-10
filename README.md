# Trojans in Alpha Centauri

This repository contains the plotting code and reduced data products associated with the forthcoming manuscript **“Planet Packing with Co-orbital Earth-mass Planets in the α Centauri System.”** The study investigates whether pairs of co-orbital, Earth-mass planets can remain stable within the habitable zone of either α Centauri A or α Centauri B. It compares systems containing only the co-orbital pair with systems perturbed by an exterior Earth-mass planet, the stellar binary companion, or both.

The repository is intentionally limited to the material needed to reproduce the manuscript figures from the supplied reduced data. Simulation-generation scripts, full simulation archives, and additional simulation data will be made available after publication.

## Repository structure

```text
Trojans-in-Alpha-Cen/
├── README.md
├── LICENSE
├── requirements.txt
├── data/                  # Reduced simulation outputs; filenames are unchanged
├── python_plot/           # One directly executable script per reproducible figure
│   └── deferred/          # Plotting scripts that require unreleased archives
└── figures/               # Figure PDFs are written here
```

## Requirements

The plotting scripts require Python 3.10 or later and the packages listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Reproducing the figures

Run the scripts from the repository root. No command-line arguments are required. Each script reads the appropriate files from `data/` and writes a PDF to `figures/`.

```bash
python python_plot/plot_Fig01.py
python python_plot/plot_Fig02.py
python python_plot/plot_Fig03.py
python python_plot/plot_Fig04.py
python python_plot/plot_Fig05.py
python python_plot/plot_Fig07.py
python python_plot/plot_Fig08.py
python python_plot/plot_Fig10.py
python python_plot/plot_Fig11.py
```

The scripts correspond to the manuscript as follows:

| Script | Manuscript figure | Content |
|---|---:|---|
| `plot_Fig01.py` | 1 | Lagrange-point and Jacobi-constant schematic |
| `plot_Fig02.py` | 2 | Beta spacing and Initial setup schematic |
| `plot_Fig03.py` | 3 | Co-orbital spacing in the control scenario |
| `plot_Fig04.py` | 4 | Co-orbital spacing with the stellar companion |
| `plot_Fig05.py` | 5 | Binary initial-mean-anomaly map |
| `plot_Fig07.py` | 7 | Exterior-planet spacing in the P3 scenario |
| `plot_Fig08.py` | 8 | Two-dimensional P3 spacing map |
| `plot_Fig09.py` | 9 | P3 initial-mean-anomaly map |
| `plot_Fig10.py` | 10 | Exterior-planet spacing in the P3+B scenario |
| `plot_Fig11.py` | 11 | Two-dimensional P3+B spacing map |

Figures 3 and 9 require REBOUND `SimulationArchive` files that were not part of the supplied pre-publication data release. Their plotting scripts are retained in `python_plot/deferred/` for use when those archives are released after publication. They are not included in the no-argument reproduction commands above because the required inputs are not currently present.

## Data files

The files in `data/` are reduced, comma-delimited simulation outputs used directly by the plotting scripts. Header lines in each file describe the host star, modeled scenario, system parameters, and column definitions.

The scenario labels used in the manuscript and scripts are:

- **C:** co-orbital planets around a single host star
- **P3:** C plus an exterior Earth-mass planet
- **B:** C plus the α Centauri stellar companion
- **P3+B:** C plus both perturbers

## Notes on reproducibility

The plotting scripts use paths relative to their own locations, so they can be launched from the repository root or from another working directory. Plot settings and output filenames are defined inside each script to keep the reproduction commands simple and explicit. The plotting code has not been converted into a Python package.

## Citation

Until the manuscript and repository receive final bibliographic information, please cite the manuscript and software repository as follows:

> Quarles, B., Nyator, F., Atuahene, C., & Lissauer, J. J. (2026). *Planet Packing with Co-orbital Earth-mass Planets in the α Centauri System*. Submitted to *The Astronomical Journal*. Code and reduced data: *Trojans in Alpha Centauri*, GitHub repository.

### Manuscript

```bibtex
@article{quarles2026trojans,
    author  = {Quarles, Billy and Nyator, Felix and Atuahene, Cavin and Lissauer, Jack J.},
    title   = {Planet Packing with Co-orbital Earth-mass Planets in the {$\alpha$} Centauri System},
    journal = {The Astronomical Journal},
    year    = {2026},
    note    = {Submitted}
}

@misc{quarles2026trojansrepository,
    author       = {Quarles, Billy and Nyator, Felix and Atuahene, Cavin and Lissauer, Jack J.},
    title        = {Trojans in Alpha Centauri: Figure Reproduction Code and Reduced Data},
    year         = {2026},
    howpublished = {GitHub repository},
    url          = {https://github.com/saturnaxis/Trojans-in-Alpha-Cen},
    note         = {Code and reduced data associated with a manuscript submitted to The Astronomical Journal}
}
```

## License

See `LICENSE` for the repository license.
