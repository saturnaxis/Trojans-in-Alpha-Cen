#!/usr/bin/env python3
"""
Plot a control-style signed-beta Hamiltonian map and overlay one simulation trajectory.

Example:
python plot_control_style_trajectory_overlay.py --archive-dir sa_binary_A --host A --scenario binary --beta0 0.400000 --perturber-ma 17.00 --output binary_A_idx00036_beta0.400000_ma017.00_control_style_overlay.png

This expects the archive-generation script's paired files:
    binary_A_idx00036_beta0.400000_ma017.00.json
    binary_A_idx00036_beta0.400000_ma017.00.bin
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import rebound
from matplotlib import rcParams

matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["savefig.dpi"] = 300

rcParams.update({'font.size': 12})
rcParams.update({'mathtext.fontset': 'cm'})


M_EARTH_IN_MSUN = 3.003e-6

STARS = {"A": {"mass": 1.0788, "luminosity": 1.5059, "radius": 1.2175, "companion": "B"}, "B": {"mass": 0.9092, "luminosity": 0.4981, "radius": 0.8591, "companion": "A"}}

DEFAULT_BETA_CRIT = {"A": 0.314, "B": 0.323}


def wrap_360_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return np.mod(angle_deg, 360.0)


def wrap_180_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def angular_difference_deg(a: float, b: float) -> float:
    return float(abs(wrap_180_deg(a - b)))


def coorbital_x(m1: float, m2: float, mstar: float) -> float:
    return 0.5 * ((m1 + m2) / (3.0 * (mstar + m1))) ** (1.0 / 3.0)


def beta_from_osculating_a(a1: float, a2: float, m1: float, m2: float, mstar: float) -> float:
    x = coorbital_x(m1, m2, mstar)
    return (a2 - a1) / (x * (a1 + a2))


def delta_lambda_geometric_hostcentric(sim: rebound.Simulation, i1: int = 1, i2: int = 2, istar: int = 0) -> float:
    ps = sim.particles
    host = ps[istar]
    x1 = ps[i1].x - host.x
    y1 = ps[i1].y - host.y
    x2 = ps[i2].x - host.x
    y2 = ps[i2].y - host.y
    theta1 = math.atan2(y1, x1)
    theta2 = math.atan2(y2, x2)
    return (theta2 - theta1) % (2.0 * math.pi)


def load_simulationarchive(path: Path):
    try:
        return rebound.Simulationarchive(str(path))
    except AttributeError:
        return rebound.SimulationArchive(str(path))


def metadata_matches(meta: dict[str, Any], host: str, scenario: str, beta0: float, perturber_ma: float | None, beta_prime: float | None, beta_tol: float, ma_tol: float, beta_prime_tol: float) -> bool:
    if meta.get("host") != host:
        return False
    if meta.get("scenario") != scenario:
        return False
    if abs(float(meta.get("beta0")) - beta0) > beta_tol:
        return False
    if scenario != "control":
        meta_ma = meta.get("perturber_ma_deg")
        if meta_ma is None or perturber_ma is None:
            return False
        if angular_difference_deg(float(meta_ma), float(perturber_ma)) > ma_tol:
            return False
    if beta_prime is not None:
        meta_bp = meta.get("beta_prime_RHm")
        if meta_bp is None:
            return False
        if abs(float(meta_bp) - beta_prime) > beta_prime_tol:
            return False
    return True


def find_archive_metadata(archive_dir: Path, host: str, scenario: str, beta0: float, perturber_ma: float | None, beta_prime: float | None, beta_tol: float, ma_tol: float, beta_prime_tol: float) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for meta_path in sorted(archive_dir.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        if not metadata_matches(meta, host, scenario, beta0, perturber_ma, beta_prime, beta_tol, ma_tol, beta_prime_tol):
            continue
        archive_path = archive_dir / meta.get("archive", meta_path.with_suffix(".bin").name)
        if not archive_path.exists():
            continue
        beta_err = abs(float(meta["beta0"]) - beta0)
        ma_err = 0.0 if scenario == "control" or perturber_ma is None else angular_difference_deg(float(meta["perturber_ma_deg"]), perturber_ma)
        score = beta_err / max(beta_tol, 1.0e-30) + ma_err / max(ma_tol, 1.0e-30)
        candidates.append((score, archive_path, meta))
    if not candidates:
        raise FileNotFoundError(f"No matching archive found in {archive_dir} for host={host}, scenario={scenario}, beta0={beta0}, perturber_ma={perturber_ma}, beta_prime={beta_prime}.")
    candidates.sort(key=lambda row: row[0])
    return candidates[0][1], candidates[0][2]


def extract_trajectory(archive_path: Path, meta: dict[str, Any]) -> dict[str, np.ndarray]:
    host = meta["host"]
    mstar = STARS[host]["mass"]
    m1 = M_EARTH_IN_MSUN
    m2 = M_EARTH_IN_MSUN
    sa = load_simulationarchive(archive_path)
    rows = {"time_yr": [], "delta_lambda_deg": [], "delta_lambda_signed_deg": [], "beta_signed": [], "a1_AU": [], "a2_AU": [], "e1": [], "e2": []}
    for sim in sa:
        ps = sim.particles
        o1 = ps[1].orbit(primary=ps[0])
        o2 = ps[2].orbit(primary=ps[0])
        beta_signed = beta_from_osculating_a(o1.a, o2.a, m1, m2, mstar)
        dlam_deg = float(wrap_360_deg(math.degrees(delta_lambda_geometric_hostcentric(sim))))
        dlam_signed_deg = float(wrap_180_deg(dlam_deg))
        rows["time_yr"].append(float(sim.t))
        rows["delta_lambda_deg"].append(dlam_deg)
        rows["delta_lambda_signed_deg"].append(dlam_signed_deg)
        rows["beta_signed"].append(float(beta_signed))
        rows["a1_AU"].append(float(o1.a))
        rows["a2_AU"].append(float(o2.a))
        rows["e1"].append(float(o1.e))
        rows["e2"].append(float(o2.e))
    return {key: np.asarray(value, dtype=float) for key, value in rows.items()}


def coorbital_hamiltonian(theta_rad: np.ndarray, beta: np.ndarray, alpha: float) -> np.ndarray:
    denom = 2.0 - 2.0 * np.cos(theta_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        potential = np.cos(theta_rad) - 1.0 / np.sqrt(denom)
        h = -1.5 * beta**2 + alpha * potential
    return np.where((denom < 1.0e-8) | (np.abs(theta_rad) < np.deg2rad(8.0)), np.nan, h)


def make_hamiltonian_grid(alpha: float, xlim: float, ylim: float, n_theta: int, n_beta: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-xlim, xlim, n_theta)
    y = np.linspace(-ylim, ylim, n_beta)
    X, Y = np.meshgrid(x, y)
    THETA = np.deg2rad(X)
    H = coorbital_hamiltonian(THETA, Y, alpha)
    return X, Y, H


def save_trajectory_csv(path: Path, traj: dict[str, np.ndarray]) -> None:
    fieldnames = list(traj.keys())
    n = len(next(iter(traj.values())))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({key: traj[key][i] for key in fieldnames})


def plot_overlay(traj: dict[str, np.ndarray], meta: dict[str, Any], archive_path: Path, alpha: float, beta_crit: float, xlim: float, ylim: float, contour_levels: int, n_theta: int, n_beta: int, cmap: str, output: Path, dpi: int, show: bool) -> None:
    X, Y, H = make_hamiltonian_grid(alpha, xlim, ylim, n_theta, n_beta)
    finite = np.isfinite(H)
    if not np.any(finite):
        raise RuntimeError("Hamiltonian grid contains no finite values.")
    h_low, h_high = np.nanpercentile(H[finite], [5.0, 99.0])
    levels = np.linspace(h_low, h_high, contour_levels)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.contour(X, Y, H, levels=levels, colors="black", linewidths=0.4, alpha=0.25)
    ax.axhline(beta_crit, color="tab:blue", linestyle="--", linewidth=1.3)
    ax.axhline(-beta_crit, color="tab:blue", linestyle="--", linewidth=1.3)
    late = traj["time_yr"] > 0.9 * np.nanmax(traj["time_yr"])
    ax.scatter(traj["delta_lambda_signed_deg"], traj["beta_signed"], s=2.0, c="black", edgecolors="none", alpha=0.2, zorder=5)
    ax.scatter(traj["delta_lambda_signed_deg"][0], traj["beta_signed"][0], s=45, c="tab:red", marker="o", edgecolors="black", linewidths=0.4, zorder=6, clip_on=False)
    #ax.scatter(np.nanmedian(traj["delta_lambda_signed_deg"][late]), np.nanmedian(traj["beta_signed"][late]), s=55, c="tab:blue", marker="D", edgecolors="black", linewidths=0.4, zorder=6)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xticks(np.arange(-180,210,30))
    ax.set_xlabel(r"$\Delta\lambda$ (deg)")
    ax.set_ylabel(r"$\beta_\mathrm{inst}$ ($R_\mathrm{H,m}$)")
    ax.grid(True, color="0.85", linewidth=0.6, alpha=0.7)
    scenario = meta.get("scenario", "unknown")
    host = meta.get("host", "unknown")
    beta0 = float(meta["beta0"])
    ma = meta.get("perturber_ma_deg", None)
    ma_text = "control" if ma is None else f"MA={float(ma):.2f}°"
    ax.text(0.99, 1.025, rf"$\alpha$ Cen {host}", transform=ax.transAxes, ha="right", fontsize='large')
    ax.text(0.99, 0.03, rf"$\beta_\mathrm{{cr}}={beta_crit:.3f}$", transform=ax.transAxes,weight='bold',color='tab:blue', ha="right", fontsize='large')
    #ax.set_title(f"{scenario}: beta0={beta0:.3f}, {ma_text}", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    print(f"wrote {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a control-style signed-beta Hamiltonian map and overlay one trajectory as black dots.")
    parser.add_argument("--archive-dir", type=Path, required=True, help="Directory containing .bin and .json archive files.")
    parser.add_argument("--host", choices=["A", "B"], required=True)
    parser.add_argument("--scenario", choices=["control", "p3", "binary", "p3_binary"], required=True)
    parser.add_argument("--beta0", type=float, required=True, help="Initial beta to match from metadata.")
    parser.add_argument("--perturber-ma", type=float, default=None, help="Perturber mean anomaly in degrees. Required except for control.")
    parser.add_argument("--beta-prime", type=float, default=None, help="Optional beta-prime filter for p3/p3_binary cases.")
    parser.add_argument("--beta-tol", type=float, default=5.0e-4, help="Tolerance for matching beta0.")
    parser.add_argument("--ma-tol", type=float, default=0.05, help="Tolerance for matching perturber mean anomaly in degrees.")
    parser.add_argument("--beta-prime-tol", type=float, default=5.0e-4, help="Tolerance for matching beta-prime.")
    parser.add_argument("--alpha", type=float, default=None, help="Hamiltonian alpha. If omitted, alpha is set from beta_crit by alpha=1.5*beta_crit^2.")
    parser.add_argument("--beta-crit", type=float, default=None, help="Critical beta for dashed guide lines. Defaults to 0.314 for A and 0.323 for B.")
    parser.add_argument("--xlim", type=float, default=180.0, help="Signed Delta-lambda half-range in degrees.")
    parser.add_argument("--ylim", type=float, default=0.4, help="Signed beta half-range.")
    parser.add_argument("--contour-levels", type=int, default=20)
    parser.add_argument("--n-theta", type=int, default=900)
    parser.add_argument("--n-beta", type=int, default=500)
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for filled Hamiltonian contours.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV path for the extracted trajectory.")
    parser.add_argument("--output", type=Path, default=Path("control_style_trajectory_overlay.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    args = parser.parse_args()
    if args.scenario != "control" and args.perturber_ma is None:
        parser.error("--perturber-ma is required for p3, binary, and p3_binary scenarios.")
    return args


def main() -> int:
    args = parse_args()
    beta_crit = args.beta_crit if args.beta_crit is not None else DEFAULT_BETA_CRIT[args.host]
    alpha = args.alpha if args.alpha is not None else 1.5 * beta_crit**2
    archive_path, meta = find_archive_metadata(args.archive_dir, args.host, args.scenario, args.beta0, args.perturber_ma, args.beta_prime, args.beta_tol, args.ma_tol, args.beta_prime_tol)
    print(f"matched archive: {archive_path}")
    print(f"matched beta0: {float(meta['beta0']):.6f}")
    if meta.get("perturber_ma_deg") is not None:
        print(f"matched perturber_ma_deg: {float(meta['perturber_ma_deg']):.6f}")
    print(f"using beta_crit={beta_crit:.6f}")
    print(f"using alpha={alpha:.6f}")
    traj = extract_trajectory(archive_path, meta)
    print(f"loaded {len(traj['time_yr'])} archive snapshots")
    print(f"time range: {traj['time_yr'][0]:.3f} to {traj['time_yr'][-1]:.3f} yr")
    print(f"signed instantaneous beta range: {np.nanmin(traj['beta_signed']):.6f} to {np.nanmax(traj['beta_signed']):.6f}")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        save_trajectory_csv(args.csv_out, traj)
        print(f"wrote {args.csv_out}")
    plot_overlay(traj, meta, archive_path, alpha, beta_crit, args.xlim, args.ylim, args.contour_levels, args.n_theta, args.n_beta, args.cmap, args.output, args.dpi, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())