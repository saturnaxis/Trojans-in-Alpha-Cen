import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound

# command used for Figure 3
# python plot_phase_space_new.py --scenario control --stride 1 --pixel-size 3.0 --marker-size 0.7 --point-alpha 0.7


SIGNED_DELTA_LAMBDA = True
PLOT_THEORY_BETA = True

M_EARTH_IN_MSUN = 3.003e-6
STARS = {"A": {"mass": 1.0788, "luminosity": 1.5059}, "B": {"mass": 0.9092, "luminosity": 0.4981}}

FS = "x-large"


def angle_difference_deg(a, b):
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def metadata_matches_ma(metadata, target_ma, ma_tol):
    if target_ma is None:
        return True

    if metadata.get("perturber_ma_deg") is None:
        return False

    return angle_difference_deg(float(metadata["perturber_ma_deg"]), float(target_ma)) <= ma_tol


def beta_theory_boundary(host):
    m1 = M_EARTH_IN_MSUN
    m2 = M_EARTH_IN_MSUN
    mstar = STARS[host]["mass"]
    mu = (m1 + m2) / (mstar + m1)

    return math.sqrt(8.0 / 3.0) * 3.0 ** (1.0 / 3.0) * mu ** (1.0 / 6.0)


def wrap180(angle_deg):
    return ((angle_deg + 180.0) % 360.0) - 180.0


def mean_longitude(orbit):
    return (orbit.Omega + orbit.omega + orbit.M) % (2.0 * math.pi)


def delta_lambda_deg(sim):
    o1 = sim.particles[1].orbit(primary=sim.particles[0])
    o2 = sim.particles[2].orbit(primary=sim.particles[0])
    lam1 = mean_longitude(o1)
    lam2 = mean_longitude(o2)
    dlam = math.degrees((lam2 - lam1) % (2.0 * math.pi))

    return wrap180(dlam) if SIGNED_DELTA_LAMBDA else dlam % 360.0


def beta_inst(sim):
    ps = sim.particles
    o1 = ps[1].orbit(primary=ps[0])
    o2 = ps[2].orbit(primary=ps[0])
    rhm = 0.5 * (o1.a + o2.a) * ((ps[1].m + ps[2].m) / (3.0 * (ps[0].m + ps[1].m))) ** (1.0 / 3.0)

    return (o2.a - o1.a) / rhm


def load_trajectory(archive_path):
    sa = rebound.Simulationarchive(str(archive_path))
    dlam_values = np.empty(len(sa))
    beta_values = np.empty(len(sa))

    for i, sim in enumerate(sa):
        dlam_values[i] = delta_lambda_deg(sim)
        beta_values[i] = beta_inst(sim)

    return dlam_values, beta_values


def load_directory(indir, stride, target_ma=None, ma_tol=1.0e-6):
    indir = Path(indir)
    meta_files = sorted(indir.glob("*.json"))
    trajectories = []
    all_dlam = []
    all_beta = []
    available_ma = []

    for meta_file in meta_files:
        metadata = json.loads(meta_file.read_text())

        if metadata.get("perturber_ma_deg") is not None:
            available_ma.append(float(metadata["perturber_ma_deg"]))

        if not metadata_matches_ma(metadata, target_ma, ma_tol):
            continue

        archive_path = indir / metadata["archive"]

        if not archive_path.exists():
            print(f"missing archive for {meta_file.name}")
            continue

        dlam, beta = load_trajectory(archive_path)
        dlam = dlam[::stride]
        beta = beta[::stride]

        trajectories.append((dlam, beta, metadata))
        all_dlam.append(dlam)
        all_beta.append(beta)

    if not trajectories:
        available_ma = sorted(set(round(ma, 6) for ma in available_ma))
        raise RuntimeError(f"No trajectories found in {indir} for perturber_ma={target_ma}. Available MA values: {available_ma}")

    return trajectories, np.concatenate(all_dlam), np.concatenate(all_beta)


def pixel_thinned_indices(ax, x, y, pixel_size):
    x = np.asarray(x)
    y = np.asarray(y)
    valid = np.isfinite(x) & np.isfinite(y)
    original_indices = np.flatnonzero(valid)

    if original_indices.size == 0:
        return original_indices

    display_coordinates = ax.transData.transform(np.column_stack((x[valid], y[valid])))
    display_cells = np.floor(display_coordinates / pixel_size).astype(np.int64)
    cell_centers = (display_cells.astype(float) + 0.5) * pixel_size
    distance_squared = np.sum((display_coordinates - cell_centers) ** 2, axis=1)

    order = np.lexsort((distance_squared, display_cells[:, 1], display_cells[:, 0]))
    sorted_cells = display_cells[order]
    first_in_cell = np.ones(len(order), dtype=bool)

    if len(order) > 1:
        first_in_cell[1:] = np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1)

    selected_indices = order[first_in_cell]

    return original_indices[np.sort(selected_indices)]


def plot_pixel_thinned_trajectories(ax, trajectories, color, pixel_size, marker_size, alpha):
    plotted_dlam = []
    plotted_beta = []
    total_samples = 0
    retained_samples = 0

    for dlam, beta, metadata in trajectories:
        valid_count = np.count_nonzero(np.isfinite(dlam) & np.isfinite(beta))
        indices = pixel_thinned_indices(ax, dlam, beta, pixel_size)

        total_samples += valid_count
        retained_samples += len(indices)

        if len(indices) == 0:
            continue

        plotted_dlam.append(dlam[indices])
        plotted_beta.append(beta[indices])

    if plotted_dlam:
        plotted_dlam = np.concatenate(plotted_dlam)
        plotted_beta = np.concatenate(plotted_beta)
        ax.scatter(plotted_dlam, plotted_beta, s=marker_size, color=color, marker="o", linewidths=0.0, alpha=alpha, edgecolors="none", rasterized=True, zorder=2)

    return retained_samples, total_samples


def format_panel(ax, sub_label, label, beta_boundary, n_trajectories, ymin, ymax):
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(-180.0, 181.0, 30.0))
    ax.set_ylabel(r"$\beta_\mathrm{inst}$ ($R_{\rm H,m}$)", fontsize=FS)
    ax.tick_params(which="major", axis="both", direction="out", length=6.0, width=2.0, labelsize="large")
    ax.tick_params(which="minor", axis="both", direction="out", length=4.0, width=1.5)
    ax.grid(alpha=0.18, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    if PLOT_THEORY_BETA:
        ax.axhline(beta_boundary, ls="--", lw=1.8, color="tab:blue", alpha=0.95, zorder=3)
        ax.axhline(-beta_boundary, ls="--", lw=1.8, color="tab:blue", alpha=0.95, zorder=3)
        ax.text(0.985, 0.055, rf"$\beta_\mathrm{{cr}}={beta_boundary:.3f}$", transform=ax.transAxes, ha="right", va="center", fontsize=FS, color="tab:blue", zorder=4)

    ax.text(0.015, 0.95, sub_label, weight="bold", transform=ax.transAxes, ha="left", va="top", fontsize=FS, zorder=4)
    ax.text(0.98, 0.965, label, transform=ax.transAxes, ha="right", va="top", fontsize=FS, zorder=4)
    #ax.text(0.98, 0.885, rf"$N_\mathrm{{traj}}={n_trajectories}$", transform=ax.transAxes, ha="right", va="top", fontsize="large", zorder=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, choices=["control", "p3", "binary", "p3_binary"], required=True)
    parser.add_argument("--indir-a", type=str, default=None)
    parser.add_argument("--indir-b", type=str, default=None)
    parser.add_argument("--outfile", type=str, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--perturber-ma", type=float, default=None)
    parser.add_argument("--ma-tol", type=float, default=1.0e-6)
    parser.add_argument("--pixel-size", type=float, default=3.0)
    parser.add_argument("--marker-size", type=float, default=0.7)
    parser.add_argument("--point-alpha", type=float, default=0.7)
    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be at least 1")

    if args.pixel_size <= 0.0:
        parser.error("--pixel-size must be greater than zero")

    if args.marker_size <= 0.0:
        parser.error("--marker-size must be greater than zero")

    if not 0.0 < args.point_alpha <= 1.0:
        parser.error("--point-alpha must be greater than zero and no greater than one")

    indir_a = args.indir_a if args.indir_a is not None else f"sa_{args.scenario}_A"
    indir_b = args.indir_b if args.indir_b is not None else f"sa_{args.scenario}_B"
    outfile = args.outfile if args.outfile is not None else f"{args.scenario}_phase_space_AB_{args.perturber_ma}_thinned.pdf"

    traj_a, dlam_a, beta_a = load_directory(indir_a, args.stride, args.perturber_ma, args.ma_tol)
    traj_b, dlam_b, beta_b = load_directory(indir_b, args.stride, args.perturber_ma, args.ma_tol)

    ymin = np.nanmin(np.concatenate((beta_a, beta_b)))
    ymax = np.nanmax(np.concatenate((beta_a, beta_b)))
    yrange = ymax - ymin
    ypad = 0.04 * yrange if yrange > 0.0 else 0.04
    plot_ymin = ymin - ypad
    plot_ymax = ymax + ypad

    beta_boundary_a = beta_theory_boundary("A")
    beta_boundary_b = beta_theory_boundary("B")

    print(f"alpha Cen A beta_crit = {beta_boundary_a:.6f}")
    print(f"alpha Cen B beta_crit = {beta_boundary_b:.6f}")
    print(f"B/A beta_crit ratio = {beta_boundary_b / beta_boundary_a:.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), dpi=300, sharex=True, sharey=True, constrained_layout=True)

    format_panel(axes[0], "a", r"$\alpha$ Cen A", beta_boundary_a, len(traj_a), plot_ymin, plot_ymax)
    format_panel(axes[1], "b", r"$\alpha$ Cen B", beta_boundary_b, len(traj_b), plot_ymin, plot_ymax)
    axes[1].set_xlabel(r"$\Delta\lambda$ (deg)", fontsize=FS)

    fig.canvas.draw()

    retained_a, total_a = plot_pixel_thinned_trajectories(axes[0], traj_a, "black", args.pixel_size, args.marker_size, args.point_alpha)
    retained_b, total_b = plot_pixel_thinned_trajectories(axes[1], traj_b, "red", args.pixel_size, args.marker_size, args.point_alpha)

    fraction_a = retained_a / total_a if total_a > 0 else 0.0
    fraction_b = retained_b / total_b if total_b > 0 else 0.0

    print(f"alpha Cen A retained {retained_a} of {total_a} samples ({fraction_a:.2%})")
    print(f"alpha Cen B retained {retained_b} of {total_b} samples ({fraction_b:.2%})")

    fig.savefig(outfile, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"wrote {outfile}")


if __name__ == "__main__":
    main()