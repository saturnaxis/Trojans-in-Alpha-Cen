import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

matplotlib.rcParams["pdf.compression"] = 9
matplotlib.rcParams["savefig.dpi"] = 300

rcParams.update({'font.size': 12})
rcParams.update({'mathtext.fontset': 'cm'})

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)


control_A = str(DATA_DIR / "results_hostA_control.txt")
control_B = str(DATA_DIR / "results_hostB_control.txt")

fs = 'x-large'

fig = plt.figure(figsize=(15, 6),dpi=150)
ax1 = fig.add_subplot(111)
# Add the zoomed inset alongside ax1 definition
ax_inset = fig.add_axes([0.45, 0.2, 0.2, 0.55]) # x, y, width, height in figure coordinates
ax_top1 = ax_inset.twiny()
ax_top2 = ax_inset.twiny()

M_A = 1.078
M_B = 0.909
M_star = [M_A, M_B]
M_E = 3.003e-6

ax = [ax1,ax_inset,ax_top1,ax_top2]
control = [control_A,control_B]
marker = ['.','x']
con_color = ['k','r']
ms = [10,5]
alpha = [1,0.5]
star_label = [r'$\alpha$ Cen A',r'$\alpha$ Cen B']
for f in range(0,2):
    data = np.genfromtxt(control[f],delimiter=",",comments="#",usecols=(4,6,7,8))
    data_outcome = np.genfromtxt(control[f],delimiter=",",comments="#",usecols=(-1),dtype='str')
    
    #plot a_crit from Dermott & Murray (1981)
    beta_crit = np.sqrt(8./3.)*3**(1./3.)*(2*M_E/M_star[f])**(1./6)
    #ax1.axvline(beta_crit,0,1,linestyle='--',color=con_color[f],lw=1.5,alpha=0.5)
    ax_inset.axvline(beta_crit,0,1,linestyle='--',color=con_color[f],lw=1.5)


    beta, delta_lam, delta_max = data[:,0],data[:,1],data[:,3]
    ax1.plot(beta,delta_lam,marker=marker[f],color=con_color[f],ms=ms[f],lw=0,label=star_label[f],alpha=alpha[f])
    col_idx = np.where(data_outcome==' collision')[0]
    ax1.plot(beta[col_idx],delta_lam[col_idx],marker=marker[f],color='b',ms=ms[f],lw=0)
    # Inset plotting logic, integrated into the loop
    # Filter data for the inset region (x=0.3-0.4)
    inset_mask = (beta >= 0.3) & (beta <= 0.4)
    ax_inset.plot(beta[inset_mask], delta_lam[inset_mask], marker=marker[f], color=con_color[f], ms=ms[f], lw=0,alpha=alpha[f])

    mask_green = delta_max < 180
    beta_min,beta_max  = 0, np.max(beta[mask_green])
    ax1.fill_betweenx(np.arange(55,186,1), beta_min,beta_max, facecolor='darkgray', alpha=0.45)
    ax_inset.fill_betweenx(np.arange(55,186,1), beta_min,beta_max, facecolor='darkgray', alpha=0.45)
    # Shaded region: delta_lam > 160 & delta_max < 355 (light yellow)
    mask_yellow = (delta_lam > 160) & (delta_max < 355)
    beta_min,beta_max  = np.max(beta[mask_green]), np.max(beta[mask_yellow])
    ax1.fill_betweenx(np.arange(55,186,1), beta_min,beta_max, facecolor='lightgray', alpha=0.35)
    ax_inset.fill_betweenx(np.arange(55,186,1), beta_min,beta_max, facecolor='lightgray', alpha=0.35)

    # # Plot collisions in the inset
    # col_idx_inset = np.where(data_outcome[inset_mask]==' collision')[0]
    # ax_inset.plot(beta[inset_mask][col_idx_inset], delta_lam[inset_mask][col_idx_inset], marker=marker[f], color='b', ms=5, lw=0)
    

for i in range(0,2):
    ax[i].minorticks_on()
    ax[i].tick_params(which='major',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    ax[i].tick_params(which='minor',axis='both', direction='out',length = 4.0, width = 3.0)
    
    beta_crit = np.sqrt(8./3.)*3**(1./3.)*(2*M_E/M_star[i])**(1./6)
    ax[i+2].set_xticks([beta_crit])
    ax[i+2].set_xlim(0.3,0.35)
    ax[i+2].set_xticklabels([r'$\beta_{\rm cr}$'],fontsize=fs,color=con_color[i])
    ax[i+2].tick_params(which='major',axis='x', direction='out',length = 6.0, width = 4.0,labelsize=fs,color=con_color[i])


ax1.legend(loc='best',fontsize=fs)

ax1.set_xlim(0,1.3)
# Inset xlim alongside main plot's
ax_inset.set_xlim(0.3, 0.35)
ax1.set_ylim(55,185)
# Inset ylim alongside main plot's
ax_inset.set_ylim(55, 185) # Keep the same y-limits as the main plot or adjust as needed
ax1.set_yticks(range(60,195,20))
# Inset xticks and yticks alongside main plot's
ax_inset.set_xticks([0.3, 0.35])
ax_inset.set_yticks([60, 100, 140, 180])

ax1.set_xlabel(r"Co-orbital Spacing $\beta$ ($R_{\rm H,m}$)",fontsize=fs)
ax1.set_ylabel(r"Average Mean Longitude" +"\n"+r"of Co-orbitals $\overline{\Delta \lambda}$ (deg.)",fontsize=fs)

fig.savefig(OUTPUT_DIR / "Fig02.pdf",bbox_inches='tight',dpi=300)
plt.close();
