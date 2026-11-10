import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib import rcParams

matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["savefig.dpi"] = 300

rcParams.update({'font.size': 12})
rcParams.update({'mathtext.fontset': 'cm'})

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


def colorbar(mappable):
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax,orientation='vertical')
    #cbar.ax.xaxis.set_ticks_position('top')
    plt.sca(last_axes)
    return cbar

#binary_data = str(DATA_DIR / "dlambda_heatmap_results.csv")
star = ["A","B"]
sub_label = ['a','b']
xi = np.arange(0,361,1)
yi = np.arange(0.04,0.51,0.01)

fs = 'x-large'
cmap = "RdBu_r"
vmin = 55.
vmax = 180.
my_cmap = matplotlib.colormaps[cmap]
norm = colors.Normalize(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))
my_cmap.set_under('w')

fig = plt.figure(figsize=(10,10),dpi=300)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax_list = [ax1,ax2]

color_label=r"Average Mean Longitude of Co-orbitals $\overline{\Delta \lambda}$ (deg.)"
cax = fig.add_axes([0.925,0.11,0.02,0.77])
cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')

cbar.set_label(color_label,fontsize=fs)
cbar.set_ticks(np.arange(60.,195,15))
cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

for i in range(0,2):
    ax = ax_list[i]
    #data = str(DATA_DIR / ("results_host%s_binary_MAmap.txt" % star[i]))
    data = str(DATA_DIR / ("results_host%s_binary_MAmap.txt" % star[i]))
    MA,f,dlambda_mean,dlambda_range = np.genfromtxt(data,delimiter=',',comments='#',usecols=(4,5,7,8),unpack=True)

    zi = griddata((MA,f),dlambda_mean,(xi[None,:],yi[:,None]),method = 'linear') #,fill_value=0
    CS = ax.pcolormesh(xi,yi,zi,cmap = cmap,vmin=vmin,vmax=vmax,shading="auto", rasterized=True, edgecolors="none", linewidth=0, antialiased=False)
    CS2 = ax.contour(xi,yi,zi,levels=[60,65,70,80,90,100],colors='w',linestyles='--')
    plt.clabel(CS2,inline=True,fontsize=12)

    if i >0:
        ax.set_xlabel(r"Initial Mean Anomaly (deg.)",fontsize=fs)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"Co-orbital Spacing $\beta$ ($R_{\rm H,m}$)",fontsize=fs)

    ax.text(0.015, 0.93, sub_label[i], color='white',weight='bold', transform=ax.transAxes, fontsize=fs)
    ax.text(0.98, 0.93, r"$\alpha$ Cen %s" % star[i], color='white',horizontalalignment='right',transform=ax.transAxes, fontsize=fs)
    ax.set_xlim(0,360)
    ax.set_xticks(np.arange(0,390,30))
    ax.set_ylim(0.04,0.5)
    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 4.0, width = 3.0)

fig.subplots_adjust(hspace=0.075)
fig.savefig(OUTPUT_DIR / "Fig05.pdf" ,bbox_inches='tight',dpi=300)
plt.close()
