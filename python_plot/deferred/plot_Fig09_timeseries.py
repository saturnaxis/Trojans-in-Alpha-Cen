#!/usr/bin/env python3
"""
Plot a signed-beta Hamiltonian phase portrait with uniformly sampled
Delta-lambda and instantaneous-beta time series.

For each requested sample time, the script asks the REBOUND
Simulationarchive for a reconstructed simulation with:

    sa.getSimulation(requested_time, mode="exact")

REBOUND starts from a nearby stored archive snapshot and integrates to the
requested time. This produces a uniform requested-time grid even when the
stored archive snapshots are slightly irregular.

Example:
python plot_resampled_phase_and_timeseries_v6.py --archive-dir sa_binary_A --host A --scenario binary --beta0 0.400000 --perturber-ma 17.00 --sample-dt 0.02 --sample-start 0 --sample-stop 250 --plot-tmax 250 --output binary_A_beta0.4_phase_timeseries.png

This expects paired .json and .bin archive files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import rebound
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["savefig.dpi"] = 300
rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "axes.linewidth": 1.1})

M_EARTH_IN_MSUN = 3.003e-6
STARS = {"A": {"mass": 1.0788, "luminosity": 1.5059, "radius": 1.2175, "companion": "B"}, "B": {"mass": 0.9092, "luminosity": 0.4981, "radius": 0.8591, "companion": "A"}}
DEFAULT_BETA_CRIT = {"A": 0.314, "B": 0.323}
SCRIPT_VERSION = "6.0"


def wrap_360_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return np.mod(angle_deg, 360.0)


def wrap_180_deg(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def angular_difference_deg(a: float, b: float) -> float:
    return float(abs(wrap_180_deg(a - b)))


def break_wrap_jumps(values: np.ndarray, jump_threshold_deg: float = 180.0) -> np.ndarray:
    """Insert NaNs after angular wrap jumps so lines do not cross the periodic boundary."""
    plotted = np.asarray(values, dtype=float).copy()
    if plotted.size > 1:
        jumps = np.abs(np.diff(plotted)) > jump_threshold_deg
        plotted[np.flatnonzero(jumps) + 1] = np.nan
    return plotted


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


def make_uniform_sample_times(archive_tmin: float, archive_tmax: float, sample_dt: float, sample_start: float | None, sample_stop: float | None) -> np.ndarray:
    """Build a uniform sampling grid contained within the archive time range."""
    if not np.isfinite(sample_dt) or sample_dt <= 0.0:
        raise ValueError(f"sample_dt must be positive and finite; got {sample_dt!r}")
    start = archive_tmin if sample_start is None else float(sample_start)
    stop = archive_tmax if sample_stop is None else float(sample_stop)
    scale = max(1.0, abs(archive_tmin), abs(archive_tmax), abs(start), abs(stop))
    tol = 64.0 * np.finfo(float).eps * scale
    if start < archive_tmin - tol:
        raise ValueError(f"sample_start={start:.16g} is before archive start tmin={archive_tmin:.16g}")
    if stop > archive_tmax + tol:
        raise ValueError(f"sample_stop={stop:.16g} is after archive end tmax={archive_tmax:.16g}")
    if stop < start:
        raise ValueError(f"sample_stop={stop:.16g} must be greater than or equal to sample_start={start:.16g}")
    start = min(max(start, archive_tmin), archive_tmax)
    stop = min(max(stop, archive_tmin), archive_tmax)
    duration = stop - start
    n_intervals = int(math.floor(duration / sample_dt + 1.0e-12))
    times = start + sample_dt * np.arange(n_intervals + 1, dtype=float)
    endpoint_tol = max(tol, 1.0e-11 * max(1.0, abs(stop)))
    if times.size and math.isclose(float(times[-1]), stop, rel_tol=0.0, abs_tol=endpoint_tol):
        times[-1] = stop
    if times.size == 0:
        raise RuntimeError("The requested sampling grid contains no times.")
    return times


def extract_trajectory(archive_path: Path, meta: dict[str, Any], sample_dt: float, sample_start: float | None, sample_stop: float | None, sampling_mode: str = "exact", progress_every: int = 1000) -> dict[str, np.ndarray]:
    """Reconstruct Delta-lambda and beta on a uniform requested-time grid."""
    if sampling_mode not in {"exact", "close"}:
        raise ValueError(f"sampling_mode must be 'exact' or 'close'; got {sampling_mode!r}")
    host_name = meta["host"]
    mstar = STARS[host_name]["mass"]
    m1 = M_EARTH_IN_MSUN
    m2 = M_EARTH_IN_MSUN
    sa = load_simulationarchive(archive_path)
    archive_tmin = float(sa.tmin)
    archive_tmax = float(sa.tmax)
    requested_times = make_uniform_sample_times(archive_tmin, archive_tmax, sample_dt, sample_start, sample_stop)
    rows = {"requested_time_yr": [], "time_yr": [], "sampling_error_yr": [], "delta_lambda_deg": [], "delta_lambda_signed_deg": [], "beta_signed": [], "a1_AU": [], "a2_AU": [], "e1": [], "e2": []}
    total = requested_times.size
    for sample_index, requested_time in enumerate(requested_times):
        sim = sa.getSimulation(float(requested_time), mode=sampling_mode)
        ps = sim.particles
        if len(ps) < 3:
            raise RuntimeError(f"Simulation at t={sim.t:.16g} yr contains fewer than the host and two co-orbital planets.")
        o1 = ps[1].orbit(primary=ps[0])
        o2 = ps[2].orbit(primary=ps[0])
        beta_signed = beta_from_osculating_a(o1.a, o2.a, m1, m2, mstar)
        dlam_deg = float(wrap_360_deg(math.degrees(delta_lambda_geometric_hostcentric(sim))))
        dlam_signed_deg = float(wrap_180_deg(dlam_deg))
        actual_time = float(sim.t)
        rows["requested_time_yr"].append(float(requested_time))
        rows["time_yr"].append(actual_time)
        rows["sampling_error_yr"].append(actual_time - float(requested_time))
        rows["delta_lambda_deg"].append(dlam_deg)
        rows["delta_lambda_signed_deg"].append(dlam_signed_deg)
        rows["beta_signed"].append(float(beta_signed))
        rows["a1_AU"].append(float(o1.a))
        rows["a2_AU"].append(float(o2.a))
        rows["e1"].append(float(o1.e))
        rows["e2"].append(float(o2.e))
        #if progress_every > 0 and (sample_index == 0 or (sample_index + 1) % progress_every == 0 or sample_index + 1 == total):
        #    print(f"resampled {sample_index + 1}/{total}: requested={requested_time:.9f} yr, actual={actual_time:.9f} yr")
    trajectory = {key: np.asarray(value, dtype=float) for key, value in rows.items()}
    if sampling_mode == "exact":
        max_error = float(np.nanmax(np.abs(trajectory["sampling_error_yr"])))
        tolerance = 1.0e-11 * max(1.0, abs(float(requested_times[0])), abs(float(requested_times[-1])))
        if max_error > tolerance:
            raise RuntimeError(f"Exact resampling did not reach the requested grid within tolerance: max |actual-requested|={max_error:.3e} yr")
    return trajectory


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


def style_axis(ax) -> None:
    ax.tick_params(which='major',axis='both', direction='out',length = 6.0, width = 4.0,labelsize='large')
    ax.tick_params(which='minor',axis='both', direction='out',length = 4.0, width = 3.0)



def add_panel_label(ax, label: str) -> None:
    ax.text(0.012, 0.975, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold", zorder=20)


def plot_phase_and_timeseries(traj: dict[str, np.ndarray], meta: dict[str, Any], alpha: float, beta_crit: float, xlim: float, ylim: float, contour_levels: int, n_theta: int, n_beta: int, plot_tmax: float | None, output: Path, dpi: int, show: bool) -> None:
    X, Y, H = make_hamiltonian_grid(alpha, xlim, ylim, n_theta, n_beta)
    xmax = float(np.nanmax(traj["time_yr"])) if plot_tmax is None else float(plot_tmax)
    finite = np.isfinite(H)
    if not np.any(finite):
        raise RuntimeError("Hamiltonian grid contains no finite values.")
    h_low, h_high = np.nanpercentile(H[finite], [5.0, 99.0])
    levels = np.linspace(h_low, h_high, contour_levels)
    fig = plt.figure(figsize=(8.5, 9.0))
    outer_grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.35, 2.0], hspace=0.24)
    lower_grid = outer_grid[1].subgridspec(nrows=2, ncols=1, height_ratios=[1.0, 1.0], hspace=0.2)
    ax_phase = fig.add_subplot(outer_grid[0])
    ax_dlam = fig.add_subplot(lower_grid[0])
    ax_beta = fig.add_subplot(lower_grid[1], sharex=ax_dlam)
    
    ax_phase.contour(X, Y, H, levels=levels, colors="black", linewidths=0.45, alpha=0.23)
    ax_phase.axhline(beta_crit, color="tab:blue", linestyle="--", linewidth=1.25)
    ax_phase.axhline(-beta_crit, color="tab:blue", linestyle="--", linewidth=1.25)
    ax_phase.scatter(traj["delta_lambda_signed_deg"], traj["beta_signed"], s=2.0, c="black", edgecolors="none", alpha=0.30, zorder=5, rasterized=True)
    ax_phase.scatter(traj["delta_lambda_signed_deg"][0], traj["beta_signed"][0], s=45, c="tab:red", marker="o", edgecolors="black", linewidths=0.4, zorder=6, clip_on=False)
    ax_phase.set_xlim(-xlim, xlim)
    ax_phase.set_ylim(-ylim, ylim)
    ax_phase.xaxis.set_major_locator(MultipleLocator(30.0))
    ax_phase.xaxis.set_minor_locator(MultipleLocator(15.0))
    ax_phase.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_phase.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax_phase.set_xlabel(r"$\Delta\lambda$ (deg)", fontsize=12)
    ax_phase.set_ylabel(r"$\beta_\mathrm{inst}$ ($R_\mathrm{H,m}$)", fontsize=12)
    #ax_phase.grid(True, which="major", color="0.86", linewidth=0.6, alpha=0.75)
    host = meta.get("host", "unknown")
    ax_phase.text(0.985, 1.025, rf"$\alpha$ Cen {host}", transform=ax_phase.transAxes, ha="right", fontsize=12)
    ax_phase.text(0.985, 0.035, rf"$\beta_\mathrm{{cr}}={beta_crit:.3f}$", transform=ax_phase.transAxes, ha="right", va="bottom", fontsize=12)
    add_panel_label(ax_phase, "a")
    style_axis(ax_phase)
    time = traj["time_yr"]
    dlam_plot = break_wrap_jumps(traj["delta_lambda_signed_deg"])
    ax_dlam.plot(time, dlam_plot, color="black", linewidth=0.9)
    ax_dlam.set_xlim(0.0, xmax)
    ax_dlam.set_ylim(0.0, 180.0)
    ax_dlam.set_yticks([0.0, 60.0, 120.0, 180.0])
    ax_dlam.yaxis.set_minor_locator(MultipleLocator(30.0))
    ax_dlam.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_dlam.set_ylabel(r"$\Delta\lambda$ (deg)", fontsize=12)
    #ax_dlam.grid(True, which="major", color="0.87", linewidth=0.6, alpha=0.75)
    ax_dlam.tick_params(labelbottom=False)
    add_panel_label(ax_dlam, "b")
    style_axis(ax_dlam)
    ax_beta.plot(time, traj["beta_signed"], color="black", linewidth=0.9)
    ax_beta.axhline(beta_crit, color="tab:blue", linestyle="--", linewidth=1.15)
    ax_beta.axhline(-beta_crit, color="tab:blue", linestyle="--", linewidth=1.15)
    #ax_beta.axhline(0.0, color="0.50", linewidth=0.75)
    ax_beta.set_xlim(0.0, xmax)
    ax_beta.set_ylim(-ylim, ylim)
    #ax_beta.yaxis.set_major_locator(MultipleLocator(0.2))
    #ax_beta.yaxis.set_minor_locator(MultipleLocator(0.1))
    #ax_beta.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax_beta.set_yticks(np.arange(-0.4,0.5,0.2))
    ax_beta.set_ylabel(r"$\beta_\mathrm{inst}$ ($R_\mathrm{H,m}$)", fontsize=12)
    ax_beta.set_xlabel("Time (yr)", fontsize=12)
    #ax_beta.grid(True, which="major", color="0.87", linewidth=0.6, alpha=0.75)
    add_panel_label(ax_beta, "c")
    style_axis(ax_beta)
    #fig.subplots_adjust(hspace=0.75)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight',dpi=dpi)
    print(f"wrote {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a signed-beta Hamiltonian map with uniformly resampled Delta-lambda and instantaneous-beta time series.")
    parser.add_argument("--archive-dir", type=Path, required=True, help="Directory containing paired .bin and .json archive files.")
    parser.add_argument("--host", choices=["A", "B"], required=True)
    parser.add_argument("--scenario", choices=["control", "p3", "binary", "p3_binary"], required=True)
    parser.add_argument("--beta0", type=float, required=True, help="Initial beta to match from metadata.")
    parser.add_argument("--perturber-ma", type=float, default=None, help="Perturber mean anomaly in degrees. Required except for control.")
    parser.add_argument("--beta-prime", type=float, default=None, help="Optional beta-prime filter for p3/p3_binary cases.")
    parser.add_argument("--beta-tol", type=float, default=5.0e-4, help="Tolerance for matching beta0.")
    parser.add_argument("--ma-tol", type=float, default=0.05, help="Tolerance for matching perturber mean anomaly in degrees.")
    parser.add_argument("--beta-prime-tol", type=float, default=5.0e-4, help="Tolerance for matching beta-prime.")
    sampling = parser.add_argument_group("uniform resampling")
    sampling.add_argument("--sample-dt", type=float, required=True, help="Requested uniform output cadence in years.")
    sampling.add_argument("--sample-start", type=float, default=None, help="First requested sample time in years. Defaults to archive tmin.")
    sampling.add_argument("--sample-stop", type=float, default=None, help="Last allowed sample time in years. Defaults to archive tmax.")
    sampling.add_argument("--sampling-mode", choices=["exact", "close"], default="exact", help="REBOUND getSimulation mode.")
    sampling.add_argument("--progress-every", type=int, default=1000, help="Print resampling progress every N samples; use 0 to disable.")
    parser.add_argument("--alpha", type=float, default=None, help="Hamiltonian alpha. If omitted, alpha=1.5*beta_crit^2.")
    parser.add_argument("--beta-crit", type=float, default=None, help="Critical beta for dashed guide lines.")
    parser.add_argument("--xlim", type=float, default=180.0, help="Signed Delta-lambda half-range in degrees.")
    parser.add_argument("--ylim", type=float, default=0.4, help="Signed beta half-range for the phase and beta panels.")
    parser.add_argument("--plot-tmax", type=float, default=250.0, help="Right edge of the time-series panels in years. Use a negative value for the last resampled time.")
    parser.add_argument("--contour-levels", type=int, default=20)
    parser.add_argument("--n-theta", type=int, default=900)
    parser.add_argument("--n-beta", type=int, default=500)
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV path for the resampled quantities.")
    parser.add_argument("--output", type=Path, default=Path("phase_delta_lambda_beta_timeseries_v6.png"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true", help="Show the plot interactively after saving.")
    args = parser.parse_args()
    if args.scenario != "control" and args.perturber_ma is None:
        parser.error("--perturber-ma is required for p3, binary, and p3_binary scenarios.")
    if args.sample_dt <= 0.0:
        parser.error("--sample-dt must be greater than zero.")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative.")
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
    print(f"requested sample_dt={args.sample_dt:.12g} yr")
    print(f"sampling mode={args.sampling_mode}")
    traj = extract_trajectory(archive_path, meta, args.sample_dt, args.sample_start, args.sample_stop, args.sampling_mode, args.progress_every)
    print(f"generated {len(traj['time_yr'])} resampled states")
    print(f"requested time range: {traj['requested_time_yr'][0]:.9f} to {traj['requested_time_yr'][-1]:.9f} yr")
    print(f"actual time range: {traj['time_yr'][0]:.9f} to {traj['time_yr'][-1]:.9f} yr")
    print(f"maximum absolute sampling error: {np.nanmax(np.abs(traj['sampling_error_yr'])):.3e} yr")
    print(f"signed instantaneous beta range: {np.nanmin(traj['beta_signed']):.6f} to {np.nanmax(traj['beta_signed']):.6f}")
    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        save_trajectory_csv(args.csv_out, traj)
        print(f"wrote {args.csv_out}")
    plot_tmax = None if args.plot_tmax < 0.0 else args.plot_tmax
    plot_phase_and_timeseries(traj, meta, alpha, beta_crit, args.xlim, args.ylim, args.contour_levels, args.n_theta, args.n_beta, plot_tmax, args.output, args.dpi, args.show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())