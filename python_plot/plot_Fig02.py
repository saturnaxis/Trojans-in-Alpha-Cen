import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath

matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["savefig.dpi"] = 300
rcParams.update({'font.size': 12})
rcParams.update({'mathtext.fontset': 'cm'})

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

host_orange = 'orange'
planet_blue_1 = '#1556C0'
planet_blue_2 = '#49A0E1'
planet_red = 'tab:red'
reference_gray = '#505960'
light_gray = '#CCD1D5'
fs = 'x-large'

fig = plt.figure(figsize=(11.2, 4.35), dpi=150)
ax1 = fig.add_axes([0.022, 0.065, 0.345, 0.895])
ax2 = fig.add_axes([0.420, 0.065, 0.565, 0.895])

for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

ax1.set_aspect('equal', adjustable='box')

# (a) Planetary geometry; orbital differences are intentionally exaggerated.
cx = 0.45
cy = 0.47
r1 = 0.215
r2 = 0.253
rmean = (r1 + r2) / 2
r3 = 0.365
th1 = np.deg2rad(0.0)
th2 = np.deg2rad(60.0)
th3 = np.deg2rad(222.492)
p1 = (cx + r1 * np.cos(th1), cy + r1 * np.sin(th1))
p2 = (cx + r2 * np.cos(th2), cy + r2 * np.sin(th2))
p3 = (cx + r3 * np.cos(th3), cy + r3 * np.sin(th3))

ax1.add_patch(Circle((cx, cy), r1, transform=ax1.transAxes, facecolor='none', edgecolor=planet_blue_1, lw=2))
ax1.add_patch(Circle((cx, cy), r2, transform=ax1.transAxes, facecolor='none', edgecolor=planet_blue_2, lw=2))
ax1.add_patch(Circle((cx, cy), rmean, transform=ax1.transAxes, facecolor='none', edgecolor='k', lw=1.5, linestyle=(0, (2.3, 3.0)), alpha=0.7))
ax1.add_patch(Circle((cx, cy), r3, transform=ax1.transAxes, facecolor='none', edgecolor=planet_red, lw=2))

# Reference ray and the counterclockwise direction of increasing longitude.
ax1.annotate('', xy=(0.945, cy), xytext=(cx, cy), xycoords=ax1.transAxes, textcoords=ax1.transAxes, arrowprops=dict(arrowstyle='-|>', color=reference_gray, linewidth=1.25, mutation_scale=12), zorder=3)
ax1.text(0.93, 0.40, r'$\lambda=0^\circ$', color=reference_gray, fontsize='large', ha='center', va='bottom', transform=ax1.transAxes)
theta = np.deg2rad(np.linspace(2, 20, 20))
r_arc = 0.45
arc_xy = np.column_stack((cx + r_arc * np.cos(theta), cy + r_arc * np.sin(theta)))
arc_arrow = FancyArrowPatch(path=MplPath(arc_xy), arrowstyle='-|>', mutation_scale=11, linewidth=1.5, color=reference_gray, transform=ax1.transAxes, zorder=5)
ax1.add_patch(arc_arrow)

# Initial mean-longitude separation of the co-orbital pair.
ax1.plot([cx, cx + 0.204 * np.cos(th2)], [cy, cy + 0.204 * np.sin(th2)], color='k', linestyle='--', lw=1.1, transform=ax1.transAxes)
ax1.add_patch(Arc((cx, cy), 0.2, 0.2, angle=0, theta1=0, theta2=60, transform=ax1.transAxes, color='k', lw=1.0))
ax1.text(0.575, 0.535, r'$\Delta \lambda_0$', color=reference_gray, fontsize='large', ha='center', transform=ax1.transAxes)

# Planetary markers and compact legend.
ax1.plot(cx, cy, marker='.', color=host_orange, ms=25, lw=0, markeredgecolor='none', transform=ax1.transAxes, zorder=9, label=r'$M_\star$')
ax1.plot(*p1, marker='.', color=planet_blue_1, ms=15, lw=0, transform=ax1.transAxes, zorder=10, label=r'$m_1$')
ax1.plot(*p2, marker='.', color=planet_blue_2, ms=15, lw=0, transform=ax1.transAxes, zorder=10, label=r'$m_2$')
ax1.plot(*p3, marker='.', color=planet_red, ms=15, lw=0, transform=ax1.transAxes, zorder=10, label=r'$m_3$')
ax1.text(0.02, 0.98, '(a)', fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax1.transAxes)
ax1.text(0.02, 0.02, 'Orbital separations exaggerated', color=reference_gray, fontsize='medium', ha='left', va='bottom', transform=ax1.transAxes)
ax1.legend(bbox_to_anchor=(0.8, 0.5, 0.3, 0.5), numpoints=1, fontsize='large', frameon=False, handletextpad=0.3)

# (b) The two normalized orbital-spacing definitions.
ax2.text(0.02, 0.98, '(b)', fontsize=fs, fontweight='bold', ha='left', va='top', transform=ax2.transAxes)
ax2.text(0.98, 0.96, 'Increasing semimajor axis', fontsize='large', color=reference_gray, ha='right', va='top', transform=ax2.transAxes)

# Upper construction: beta measures the spacing between the co-orbital planets.
x1 = 0.165
x2 = 0.410
y1 = 0.765
ax2.annotate('', xy=(0.968, y1), xytext=(0.090, y1), xycoords=ax2.transAxes, textcoords=ax2.transAxes, arrowprops=dict(arrowstyle='->', color=light_gray, linewidth=1.4, mutation_scale=11))
ax2.plot([x1, x1], [y1 - 0.055, y1 + 0.055], color=planet_blue_1, lw=2, transform=ax2.transAxes)
ax2.plot([x2, x2], [y1 - 0.055, y1 + 0.055], color=planet_blue_2, lw=2, transform=ax2.transAxes)
ax2.text(x1, y1 + 0.087, r'$a_1$', color=planet_blue_1, fontsize=fs, ha='center', transform=ax2.transAxes)
ax2.text(x2, y1 + 0.087, r'$a_2$', color=planet_blue_2, fontsize=fs, ha='center', transform=ax2.transAxes)
ax2.plot([x1, x1], [y1 - 0.065, 0.649], color=planet_blue_1, linestyle=':', lw=1.5, transform=ax2.transAxes)
ax2.plot([x2, x2], [y1 - 0.065, 0.649], color=planet_blue_2, linestyle=':', lw=1.5, transform=ax2.transAxes)
ax2.annotate('', xy=(x2, 0.649), xytext=(x1, 0.649), xycoords=ax2.transAxes, textcoords=ax2.transAxes, arrowprops=dict(arrowstyle='<->', color=planet_blue_1, lw=2, mutation_scale=11))
ax2.text((x1 + x2) / 2, 0.673, r'$\beta$', color=planet_blue_1, fontsize=fs, ha='center', va='bottom', transform=ax2.transAxes)
ax2.text(0.691, 0.657, r'$\beta=\dfrac{a_2-a_1}{R_{\rm H,m}}$', color=planet_blue_1, fontsize=fs, ha='center', va='center', transform=ax2.transAxes)

# Lower construction: beta* is measured from the pair's mean orbit to m3.
x_mean = 0.286
x3 = 0.835
y2 = 0.430
ax2.annotate('', xy=(0.968, y2), xytext=(0.090, y2), xycoords=ax2.transAxes, textcoords=ax2.transAxes, arrowprops=dict(arrowstyle='->', color=light_gray, lw=1.4, mutation_scale=11))
ax2.plot([x_mean, x_mean], [y2 - 0.055, y2 + 0.055], color='k', lw=2, transform=ax2.transAxes)
ax2.plot([x3, x3], [y2 - 0.055, y2 + 0.055], color=planet_red, lw=2, transform=ax2.transAxes)
ax2.text(x_mean, y2 + 0.089, r'$\tilde a_{12}$', color='k', fontsize=fs, ha='center', transform=ax2.transAxes)
ax2.text(x3, y2 + 0.089, r'$a_3$', color=planet_red, fontsize=fs, ha='center', transform=ax2.transAxes)
ax2.plot([x_mean, x_mean], [y2 - 0.063, 0.297], color='k', lw=1.5, linestyle=':', transform=ax2.transAxes)
ax2.plot([x3, x3], [y2 - 0.063, 0.297], color=planet_red, lw=1.5, linestyle=':', transform=ax2.transAxes)
ax2.annotate('', xy=(x3, 0.297), xytext=(x_mean, 0.297), xycoords=ax2.transAxes, textcoords=ax2.transAxes, arrowprops=dict(arrowstyle='<->', color=planet_red, lw=2, mutation_scale=11))
ax2.text((x_mean + x3) / 2, 0.324, r'$\beta^*$', color=planet_red, fontsize=16, ha='center', va='bottom', transform=ax2.transAxes)

# Both lower definitions occupy one horizontal line, parallel to the beta* arrow.
ax2.text(0.42, 0.12, r'$\beta^*=\dfrac{a_3-\tilde a_{12}}{R_{\rm H,m}^*}$', color=planet_red, fontsize=fs, ha='center', va='center', transform=ax2.transAxes)
ax2.text(0.70, 0.12, r'$\tilde a_{12}=\dfrac{a_1+a_2}{2}$', color='k', fontsize=fs, ha='center', va='center', transform=ax2.transAxes)

fig.savefig(OUTPUT_DIR / 'Fig02.pdf', bbox_inches='tight', dpi=300)
plt.close(fig)
