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


def potential(x,y):
	ro = np.sqrt((x+mu)**2+y**2)
	r1 = np.sqrt((x-(1.-mu))**2+y**2)
	U = (1.-mu)/ro + mu/r1 + (x**2+y**2)/2.
	return 2.*U 

mu = 1./81.
xi = np.arange(-2.,2.,0.001)
yi = np.arange(-2.,2.,0.001)
xx,yy = np.meshgrid(xi,yi)
zi = potential(xx,yy)
L1 = (mu/3.)**(1./3)
L3 = -(1.+(5*mu/12.))
L4 = 0.5-mu
col = 'k'
mark = '.'
ms = 15

fig  = plt.figure(figsize=(6,6),dpi=150)
ax = fig.add_subplot(111,aspect='equal')

#ax.plot((1.-mu)*np.cos(theta),(1.-mu)*np.sin(theta),'k-',lw=2)
CS = ax.contour(xi,yi,zi,np.arange(3,3.2,0.05),cmap=plt.cm.Set1)
ax.plot([-1.25,1.35],[0,0],'k--',ms=1.5)
ax.plot(1.-L1,0,marker=mark,color=col,ms=ms)
ax.text(1.-L1,0.065,'$L_1$',fontsize=20,horizontalalignment='center')
ax.plot(1.+L1,0,marker=mark,color=col,ms=ms)
ax.text(1.+L1+0.025,0.065,'$L_2$',fontsize=20,horizontalalignment='center')
ax.plot(L3+mu,0,marker=mark,color=col,ms=ms)
ax.text(L3-0.1+mu,0.065,'$L_3$',fontsize=20,horizontalalignment='center')
ax.plot(L4,np.sqrt(3)/2.,marker=mark,color=col,ms=ms)
ax.text(L4,np.sqrt(3)/2.+0.065,'$L_4$',fontsize=20,horizontalalignment='center')
ax.plot(L4,-np.sqrt(3)/2.,marker=mark,color=col,ms=ms)
ax.text(L4,-np.sqrt(3)/2.-0.135,'$L_5$',fontsize=20,horizontalalignment='center')

ax.plot([L4,-mu],[np.sqrt(3.)/2.,0],'k--',lw=1.5)
ax.plot([L4,1.-mu],[np.sqrt(3.)/2.,0],'k--',lw=1.5)
ax.plot([L4,-mu],[-np.sqrt(3.)/2.,0],'k--',lw=1.5)
ax.plot([L4,1.-mu],[-np.sqrt(3.)/2.,0],'k--',lw=1.5)

ax.plot(-mu,0,'.',mfc='orange',mec='orange',ms=50,label='Host Star')
ax.plot(1.-mu,0,'.',mfc='darkblue',mec='darkblue',ms=20,label='Planet')
#ax.plot(0,0,'kx',ms=5,label="CoM")

ax.legend(bbox_to_anchor=(1.1, .5, .3, .5),numpoints=1,fontsize=20,frameon=False)
ax.set_xlim(-1.3,1.3)
ax.set_ylim(-1.3,1.3)
ax.axis('off')

fig.savefig(OUTPUT_DIR / "Fig01.pdf",bbox_inches='tight',dpi=300)
plt.close()
