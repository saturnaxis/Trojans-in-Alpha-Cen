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
    cbar = fig.colorbar(mappable, cax=cax,orientation='horizontal')
    cbar.ax.xaxis.set_ticks_position('top')
    plt.sca(last_axes)
    return cbar

def beta_period_ratio(p,M_star,order):
    #return the value of beta given a period ratio and stellar mass M_star
    PR = p/(p-order)
    X = 0.5*((3*M_E)/(3*(M_star+2*M_E)))**(1./3)
    return (PR**(2./3) - 1)/(PR**(2./3)+1)*(1/X)

M_A = 1.078
M_B = 0.909
M_star = [M_A, M_B]
M_E = 3.003e-6

star = ["A","B"]
sub_label = ['a','b']

xi = np.arange(3.5,11,0.01)
yi = np.arange(0.04,1.305,0.005)

fs = 'x-large'
cmap = "RdBu_r"
vmin = 55.
vmax = 180.
my_cmap = matplotlib.colormaps[cmap]
norm = colors.Normalize(vmin,vmax)
cmmapable =cm.ScalarMappable(norm,my_cmap)
cmmapable.set_array(range(0,1))
my_cmap.set_under('w')

fig = plt.figure(figsize=(10,10),dpi=150)
ax1 = fig.add_subplot(211)
ax_top1 = ax1.twiny()
ax2 = fig.add_subplot(212)
ax_top2 = ax2.twiny()

ax_list = [ax1,ax2]
ax_top_list = [ax_top1,ax_top2]

color_label=r"Average Mean Longitude of Co-orbitals $\overline{\Delta \lambda}$ (deg.)"
cax = fig.add_axes([0.92,0.11,0.02,0.77])
cbar=plt.colorbar(cmmapable,cax=cax,orientation='vertical')
cbar.solids.set_rasterized(True)
cbar.set_label(color_label,fontsize=fs)
cbar.set_ticks(np.arange(60.,195,15))
cbar.ax.tick_params(axis='both', direction='out',length = 8.0, width = 8.0,labelsize=fs)

pr_labels1 = ["%i:%i" % (p,p-1) for p in range(13,4,-1)]
pr_labels2 = ["%i:%i" % (p,p-2) for p in range(19,9,-2)]
pr_labels = np.concatenate((pr_labels1,pr_labels2))

for i in range(0,2):
    ax = ax_list[i]
    ax_top = ax_top_list[i]
    data = str(DATA_DIR / ("results_host%s_control_p3_binary_2D.txt" % star[i]))
    f,beta,dlambda_mean,dlambda_range = np.genfromtxt(data,delimiter=',',comments='#',usecols=(4,5,6,7),unpack=True)

    zi = griddata((beta,f),dlambda_mean,(xi[None,:],yi[:,None]),method = 'linear') #,fill_value=0
    CS = ax.pcolormesh(xi,yi,zi,cmap = cmap,vmin=vmin,vmax=vmax,shading="auto", rasterized=True)
    #CS2 = ax.contour(xi,yi,zi,levels=[60,65,70,80,90,100],colors='w',linestyles='--')
    #plt.clabel(CS2,inline=True,fontsize=12)

    beta_pr1 = beta_period_ratio(np.arange(13,4,-1),M_star[i],1)
    beta_pr2 = beta_period_ratio(np.arange(19,9,-2),M_star[i],2)
    beta_pr = np.concatenate((beta_pr1,beta_pr2))
    ax_top.set_xticks(beta_pr)
    ax_top.set_xticklabels(pr_labels,rotation=90,fontsize=fs)
    ax_top.tick_params(which='major',axis='x', direction='out',length = 6.0, width = 4.0,labelsize='medium',color='tab:blue')

    if i > 0:
        ax.set_xlabel(r"Spacing of External Planet $\beta^\prime$ ($R_{\rm H,m}^\prime$)",fontsize=fs)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(r"Co-orbital Spacing $\beta$ ($R_{\rm H,m}$)",fontsize=fs)
    ax.text(0.015, 0.93, sub_label[i], color='white',weight='bold', transform=ax.transAxes, fontsize=fs)
    ax.text(0.98, 0.93, r"$\alpha$ Cen %s" % star[i], color='white',horizontalalignment='right',transform=ax.transAxes, fontsize=fs)
    
    ax.set_xlim(3.5,11)
    ax_top.set_xlim(3.5,11)
    ax.set_ylim(0.04,0.55)
    ax.minorticks_on()
    ax.tick_params(which='major',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    ax.tick_params(which='minor',axis='both', direction='out',length = 4.0, width = 3.0)

fig.subplots_adjust(hspace=0.25)
fig.savefig(OUTPUT_DIR / "Fig11.pdf",bbox_inches='tight',dpi=300)
plt.close()
