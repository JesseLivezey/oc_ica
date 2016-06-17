from __future__ import division
import pdb,h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import utils
reload(utils)
from utils import tile_raster_images
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

class FixedOrderFormatter(mpl.ticker.ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        mpl.ticker.ScalarFormatter.__init__(self, useOffset=useOffset, 
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

def plot_colorbar(pos,n,label,ticklabels,cmap=None):
    cax = plt.axes(pos)
    col = np.linspace(0,1,n)
    if cmap is None:
       cmap = cm.viridis
    for i in xrange(n):
            cax.add_patch(plt.Rectangle((i,0),1,1,\
                    facecolor=cmap(col[i]),edgecolor=cmap(col[i])))
    cax.set_xlim(0,i+1)
    cax.xaxis.tick_top()
    cax.set_yticks([])
    cax.set_xticks([.5,i+.5])
    cax.xaxis.label_position='top'
    cax.set_xticklabels(ticklabels,fontsize=12,stretch='ultra-condensed')
    #cax.xaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext())
    cax.set_xlabel(label,labelpad=20,fontsize=20)
    return cax

def plot_error_sparsity(error,sparsity,labels,savePath=None):
    """
    Makes error-sparsity plots
    
    Parameters:
    -----------
    
    error: array,
            dimensions: n_degeneracy_controls x n_lambdas
    sparsity: array,
            dimensions: n_degeneracy_controls x n_lambdas
    lambd: array,
            lambda values
    labels: list of strings,
            length : n_degeneracy_controls
    colors: list of strings,
            length : n_degeneracy_controls
    """
    error = np.atleast_2d(error)
    sparsity = np.atleast_2d(sparsity)
    n = sparsity.shape[0]
    fig = plt.figure('error_sparsity')
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    col = np.linspace(0,1,sparsity.shape[-1])
    colors = ['r','g','b','k']
    for i in xrange(n):
        ax.plot(sparsity[i],error[i],label=labels[i],color=colors[i],lw=3,alpha=.2)
    ax.legend(loc='best', frameon=False,fontsize=16,ncol=1)
    cmaps = ['Reds','Greens','Blues','Greys']
    for i in xrange(n):
        ax.scatter(x=sparsity[i],y=error[i],c=col,marker='o',label=labels[i],\
                    cmap=cmaps[i],s=80)
    ax.set_xlabel(r'$\sum_{i}\sum_{j}\mathrm{logcosh}(W_jx^{(i)})$',fontsize=18,labelpad=2)
    ax.set_ylabel(r'$||\frac{X}{||X||_2}-\frac{\hat{X}}{||\hat{X}||_2}||_{2}^2$',fontsize=18,labelpad=2)
    #ax.set_ylim(-1,100)
    ax.set_yscale('log')
    """
    cax_pos = [0.185,0.45,0.30,0.025]
    cax_ticklabels = [lambd[0],lambd[-1]]
    cax_label = r'$\lambda$'
    cax = plot_colorbar(cax_pos,len(lambd),cax_label,cax_ticklabels,cmap=cm.Greys)
    """
    #ax.xaxis.set_major_formatter(FixedOrderFormatter(-2))
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_angle_hist(angles,sparsity,savePath=None):
    """
    Plots angle histograms
    
    Parameters:
    -----------
    
    angles: array
            dimensions: n_sparsity_values x n_angles
    sparsity:  array
            sparsity values
    """
    fig = plt.figure('angle_hist')
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    col = np.linspace(0,1,len(sparsity))
    for i in xrange(len(sparsity)):
        h,b = np.histogram(angles[i],100)
        ax.plot(b[:-1],h,drawstyle='steps-mid',color=cm.viridis(col[i]),lw=2,alpha=.4)
    ax.set_yscale('log')
    ax.set_xlim(np.min(angles),90)
    cax_pos = [0.285,0.65,0.30,0.025]
    cax_ticklabels = [sparsity[0],sparsity[-1]]
    cax_label = r'sparsity'
    cax = plot_colorbar(cax_pos,len(sparsity),cax_label,cax_ticklabels)
    #ax.legend(loc='best', frameon=False,fontsize=12,ncol=3)
    ax.set_xlabel(r'$\theta$',fontsize=20)
    ax.set_ylabel('counts',fontsize=20)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()


def plot_angle_hist_labels(angles,labels,density=True,\
                           savePath=None,ax=None):
    """
    Plots angle histograms
    
    Parameters:
    -----------
    
    angles: array
            dimensions: n_sparsity_values x n_angles
    sparsity:  array
            sparsity values
    """
    if ax is None:
        fig = plt.figure('angle_hist',figsize=(4,4))
        fig.clf()
        ax = plt.axes([.15,.1,.8,.8])
    col = np.linspace(0,1,len(labels))
    for i in xrange(len(labels)):
        h,b = np.histogram(angles[i],np.arange(0,91))
        if density:
             h= h/np.sum(h)
        ax.plot(b[:-1],h,drawstyle='steps',color=cm.viridis(col[i]),lw=1.5,label=labels[i])
    ax.set_yscale('log')
    if not density:
        ax.set_ylabel('Counts',fontsize=16)
        ax.set_yticks([1e0,1e2,1e4])
    else:
        ax.set_ylabel('Density',fontsize=16,labelpad=-10)
        ax.set_yticks([1e-5,1e0])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    #ax.set_xlim(np.min(angles),90)
    ax.set_xlim(20,90)
    ax.legend(loc='best', frameon=False,fontsize=12,ncol=1)
    ax.set_xlabel(r'$\theta$',fontsize=16,labelpad=0)
    ax.set_xticks([20,55,90])
    
    if savePath is not None:
        plt.savefig(savePath,dpi=300)


def plot_norms(mode=0,savePath=None):
    import matplotlib as mpl
    formatter=mpl.ticker.FormatStrFormatter('%.1f')
    fig = plt.figure('norms',figsize=(3,3))
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    norms = ['L2', 'COULOMB', 'RANDOM', 'L4']
    norms = [r'$L_2$', 'Coulomb', 'Random prior', r'$L_4$']
    col = np.linspace(0,1,len(norms))
    if mode==0:
        xx = np.linspace(0,1,100)
        fun = [lambda x: x**2,lambda x:(1.1-x**2)**(-1/2.),lambda x:-np.log(1-x**2),lambda x:x**4]
    elif mode==1:
        xx = np.linspace(-.17,.17,100)
        fun = [lambda x: -2*x,lambda x:-x/(1-x**2)**(3/2),lambda x:-(2*x)/(1-x**2),lambda x:-4*x**3] 
    for i in xrange(4):
        ax.plot(xx,fun[i](xx),color=cm.viridis(col[i]),lw=2,label=norms[i])
    if mode==1:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(np.min(xx),np.max(xx))
    if mode==0:
        ax.legend(loc='upper left', frameon=False,fontsize=10,ncol=1)
        ax.set_ylim(0,3)
        ax.set_xticks(np.arange(0,1.1,.25))
        ax.set_yticks(np.arange(0,3.1,1.))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$C(\cos\,\theta)$',fontsize=12,labelpad=0)
        ax.set_xlabel(r'$\cos\,\theta$',fontsize=12)
    elif mode==1:
        ax.set_ylim(-.2,.2)
        ax.set_xticks(np.arange(-.15,.16,.15))
        ax.set_yticks(np.arange(-.2,.21,.1))
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',fontsize=12,labelpad=65)#20)
        ax.set_xlabel(r'$\cos\,\theta$',fontsize=12,labelpad=80)#180)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)

def plot_angle_diff_hist(angles_null, angles, sparsity,savePath=None):
    """
    Plots angle histograms
    
    Parameters:
    -----------
    
    angles: array
            dimensions: n_lambdas x n_angles
    lambd:  array
            lambda values
    """
    fig = plt.figure('angle_hist')
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    col = np.linspace(0,1,len(sparsity))
    for i in xrange(len(sparsity)):
        h_null,b_null = np.histogram(angles_null[i], range(0, 91))
        h,b = np.histogram(angles[i], range(0, 91))
        ax.plot(b[:-1],h-h_null,drawstyle='steps-mid',color=cm.viridis(col[i]),lw=2,alpha=.4)
    ax.set_xlim(np.min(angles),90)
    cax_pos = [0.285,0.65,0.30,0.025]
    cax_ticklabels = [sparsity[0],sparsity[-1]]
    cax_label = r'sparsity'
    cax = plot_colorbar(cax_pos,len(sparsity),cax_label,cax_ticklabels)
    #ax.legend(loc='best', frameon=False,fontsize=12,ncol=3)
    ax.set_xlabel(r'$\theta$',fontsize=20)
    ax.set_ylabel('counts',fontsize=20)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)

def plot_filters(filters,n_pixels=8,n_filters=16,\
                 savePath=None,ax=None):
    if ax is None:
        fig = plt.figure('filters')
        fig.clf()
        ax = plt.axes()
    im = tile_raster_images(filters,(n_pixels,n_pixels),(n_filters,n_filters),
                       (2,2),scale_rows_to_unit_interval=False,
                       output_pixel_vals=False)
    ax.imshow(im,aspect='auto',interpolation='nearest',cmap='gray')
    ax.set_axis_off()
    if savePath is not None:
        plt.savefig(savePath,dpi=300)

def plot_figure3(fileName,oc=4,idx=7,savePath=None):
    f = h5py.File(fileName,'r')
    labels = ['L2','COULOMB','RANDOM','L4']
    angles = np.zeros((4,len(f['oc_%i/%s/angles'%(oc,labels[0])][idx])))
    for i,key in enumerate(labels):
        angles[i] = f['oc_%i/%s/angles'%(oc,key)][idx]
    fig = plt.figure('Figure3',figsize=(6,3))     
    fig.clf()
    ax_angles = plt.axes([.125,.15,.35,.7])
    labels = [r'$L_2$','Coulomb','Random prior',r'$L_4$']
    plot_angle_hist_labels(angles,labels,density=True,ax=ax_angles)
    ax_bases = plt.axes([.55,.15,.4,.8])
    plot_filters(f['oc_%i/L4/wbases'%(oc)][idx],ax=ax_bases)
    fig.text(.01,.9,'a)',fontsize=14)
    fig.text(.5,.9,'b)',fontsize=14)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)

