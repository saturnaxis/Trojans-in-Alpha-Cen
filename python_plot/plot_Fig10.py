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


control_A = str(DATA_DIR / "results_hostA_control_p3_binary_2D.txt")
control_B = str(DATA_DIR / "results_hostB_control_p3_binary_2D.txt")

def beta_period_ratio(p,M_star):
    #return the value of beta given a period ratio and stellar mass M_star
    PR = p/(p-1)
    X = 0.5*((3*M_E)/(3*(M_star+2*M_E)))**(1./3)
    return (PR**(2./3) - 1)/(PR**(2./3)+1)*(1/X)

def period_ratio(beta,M_star):
    #return the period ratio given a beta and stellar mass M_star
    X = 0.5*((3*M_E)/(3*(M_star+2*M_E)))**(1./3)
    return ((1+beta*X)/(1-beta*X))**1.5

M_A = 1.078
M_B = 0.909
M_star = [M_A, M_B]
M_E = 3.003e-6

fs = 'x-large'

fig = plt.figure(figsize=(15, 6),dpi=150)
ax1 = fig.add_subplot(111)
'''ax1_top_A = ax1.twiny()
ax1_top_A.set_xlim(3.5,11)
ax1_top_B = ax1.twiny()
ax1_top_B.set_xlim(3.5,11)'''

# Add the zoomed inset alongside ax1 definition
ax_inset = fig.add_axes([0.3, 0.5, 0.5, 0.25]) # x, y, width, height in figure coordinates
ax1ins_top_A = ax_inset.twiny()
ax1ins_top_B = ax_inset.twiny()


ax = [ax1,ax_inset]
#ax_top = [ax1_top_A,ax1_top_B]
ax_top_ins = [ax1ins_top_A,ax1ins_top_B]
control = [control_A,control_B]
marker = ['.','x']
con_color = ['k','r']
ms = [10,5]
alpha = [1,0.5]
star_label = [r'$\alpha$ Cen A',r'$\alpha$ Cen B']
pr_labels = ["%i:%i" % (p,p-1) for p in range(10,4,-1)]
for f in range(0,2):
    data = np.genfromtxt(control[f],delimiter=",",comments="#",usecols=(4,5,6,7,8))
    data_outcome = np.genfromtxt(control[f],delimiter=",",comments="#",usecols=(-1),dtype='str')
    data_idx = np.where((data[:,0]-np.min(data[:,0]))<1e-4)[0]
    data = data[data_idx]
    data_outcome = data_outcome[data_idx]

    beta, delta_lam, delta_max = data[:,1],data[:,2],data[:,-4]
    ax1.plot(beta,delta_lam,marker=marker[f],color=con_color[f],ms=5,lw=0,label=star_label[f],alpha=alpha[f])
    col_idx = np.where(data_outcome==' collision')[0]
    ax1.plot(beta[col_idx],delta_lam[col_idx],marker=marker[f],color='b',ms=5,lw=0)
    # Inset plotting logic, integrated into the loop
    # Filter data for the inset region (x=0.3-0.4)
    inset_mask = beta >= 3.8
    ax_inset.plot(beta[inset_mask], delta_lam[inset_mask], marker=marker[f], color=con_color[f], ms=5, lw=0,alpha=alpha[f])
    ax_inset.plot(beta[col_idx],delta_lam[col_idx],marker=marker[f],color='b',ms=5,lw=0)
    beta_pr = beta_period_ratio(np.arange(10,4,-1),M_star[f])
    #ax_top[f].set_xticks(beta_pr)
    ax_top_ins[f].set_xticks(beta_pr)
    #ax_top[f].set_xticklabels(pr_labels,rotation=90,fontsize=fs,color=con_color[f])
    ax_top_ins[f].set_xticklabels(pr_labels,rotation=90,fontsize=fs,color=con_color[f])
    #ax_top[f].tick_params(which='major',axis='x', direction='out',length = 6.0, width = 4.0,labelsize=fs,color=con_color[f])
    ax_top_ins[f].tick_params(which='major',axis='x', direction='out',length = 6.0, width = 4.0,labelsize=fs,color=con_color[f])
    for b in beta_pr:
        ax_inset.axvline(b,color=con_color[f],linestyle='--',lw=1.5)

for i in range(0,2):
    ax[i].minorticks_on()
    ax[i].tick_params(which='major',axis='both', direction='out',length = 6.0, width = 4.0,labelsize=fs)
    ax[i].tick_params(which='minor',axis='both', direction='out',length = 4.0, width = 3.0)


ax1.legend(loc='upper left',fontsize=fs)

ax1.set_xlim(3.5,11)
ax_inset.set_xlim(3.7,11)
ax_inset.set_ylim(55,75)
ax1.set_ylim(55,185)
ax1ins_top_A.set_xlim(3.7,11)
ax1ins_top_B.set_xlim(3.7,11)

#ax_inset.set_ylim(55, 185) # Keep the same y-limits as the main plot or adjust as needed
ax1.set_yticks(range(60,195,20))
# Inset xticks and yticks alongside main plot's
ax_inset.set_xticks(range(4,11,1))
#ax_inset.set_yticks([60, 100, 140, 180])

ax1.set_xlabel(r"Spacing of External Planet $\beta^*$ ($R_{\rm H,m}^*$)",fontsize=fs)
ax1.set_ylabel(r"Average Mean Longitude" +"\n"+r"of Co-orbitals $\overline{\Delta \lambda}$ (deg.)",fontsize=fs)

fig.savefig(OUTPUT_DIR / "Fig10.pdf",bbox_inches='tight',dpi=300)
plt.close();
