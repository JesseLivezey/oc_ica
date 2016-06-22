from __future__ import division
import pdb,h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches

mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

import utils
reload(utils)
from utils import tile_raster_images as tri
from utils import fractional_polar_axes as polar
import evaluate_degeneracy_controls as dgcs
reload(dgcs)
from optimizers import adam
import ica as ocica
reload(ocica)
import datasets as ds
reload(ds)
import gabor_fit as fit
reload(fit)

def plot_costs(mode=0,savePath=None):
    formatter=mpl.ticker.FormatStrFormatter('%.1f')
    fig = plt.figure('costs',figsize=(3,3))
    fig.clf()
    ax = plt.axes([.16,.15,.8,.81])
    costs = [r'$L_2$', 'Coulomb', 'Random prior', r'$L_4$']
    col = np.linspace(0,1,len(costs))
    if mode==0:
#        xx = np.linspace(0,1,100)
#        fun = [lambda x: x**2,lambda x:(1.1-x**2)**(-1/2.),lambda x:-np.log(1-x**2),lambda x:x**4]
        xx = np.linspace(.6,1.,100)
        fun = [lambda x: 2*x,lambda x: x/(1-x**2)**(3/2),lambda x: (2*x)/(1-x**2),lambda x: 4*x**3] 
        ax.set_yscale('log')
    elif mode==1:
        xx = np.linspace(-.17,.17,100)
        fun = [lambda x: 2*x,lambda x: x/(1-x**2)**(3/2),lambda x: (2*x)/(1-x**2),lambda x: 4*x**3] 
#        fun = [lambda x: -2*x,lambda x:-x/(1-x**2)**(3/2),lambda x:-(2*x)/(1-x**2),lambda x:-4*x**3]
    for i in xrange(4):
        ax.plot(xx,fun[i](xx),color=cm.viridis(col[i]),lw=2,label=costs[i])
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
        #ax.set_ylim(0,3)
        #ax.set_xticks(np.arange(0,1.1,.25))
        #ax.set_yticks(np.arange(0,3.1,1.))
        ax.set_ylim(1e0,1e2)
        ax.set_xticks(np.arange(0.6,1.1,.2))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',fontsize=12,labelpad=-4)#20)
#        ax.set_ylabel(r'$C(\cos\,\theta)$',fontsize=12,labelpad=0)
        ax.set_xlabel(r'$\cos\,\theta$',fontsize=12)
    elif mode==1:
        ax.set_ylim(-.2,.2)
        ax.set_xticks(np.arange(-.15,.16,.15))
        ax.set_yticks(np.arange(-.2,.21,.1))
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',fontsize=12,labelpad=65)#20)
        ax.set_xlabel(r'$\cos\,\theta$',fontsize=12,labelpad=80)#180)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)

def plot_angles_1column(W,W_0,costs,cmap=plt.cm.viridis,
                        savePath=None,density=True):
    col = np.linspace(0,1,W.shape[1])
    if W.shape[0]>1:
        figsize=(6,3)
    else:
        figsize=(4,4)
    fig = plt.figure('angle distributions',figsize=figsize)
    columns = W.shape[0]
    rows = W.shape[1]
    count = 1
    for i in xrange(columns):
        ax = fig.add_subplot(columns,1,i+1)
        for j in xrange(rows):
            count+=1
            angles = dgcs.compute_angles(W[i,j])
            h,b = np.histogram(angles,np.arange(0,91)) 
            if density:
                h = h*1./np.sum(h)
            b = np.arange(1,91)
            ax.plot(b,h,drawstyle='steps-pre',color=cmap(col[j]),
                    lw=1.5,label=costs[j])

            angles0 = dgcs.compute_angles(W_0[i])
            h0,b0 = np.histogram(angles0,np.arange(0,91))
            if density:
                h0 = h0*1./np.sum(h0)
            b0 = np.arange(1,91)
            ax.plot(b0,h0,'--',drawstyle='steps-pre',color='r',lw=1)
            ax.set_yscale('log')
            ax.set_xlim(0,90) 
            
            if density:
                ax.set_ylim(1e-4,1e0)
            else:
                ax.set_ylim(1e0,1e4) 

            if W.shape[0]>1:
                if j==0:
                    fig.text(.965,.9-.3*i,initial_conditions[i],
                            rotation='vertical',fontsize=16)
                if i==0:
                    ax.legend(loc='upper left', frameon=False,fontsize=12,ncol=1)
                if i==2:
                    ax.set_xlabel(r'$\theta$',fontsize=16)
                    if density:
                        ax.set_ylabel('Density',fontsize=16)
                    else:
                        ax.set_ylabel('Counts',fontsize=16)
                else:
                    ax.set_yticklabels([])
                    ax.set_xticklabels([])
                ax.set_xticks([0,45,90])

            else:
                ax.legend(loc='upper left', frameon=False,fontsize=12,ncol=1)
                ax.set_xlabel(r'$\theta$',fontsize=16,labelpad=-10)
                if density:
                    ax.set_ylabel('Density',fontsize=16,labelpad=2)
                else:
                    ax.set_ylabel('Counts',fontsize=16)
                ax.set_xlim(45,90)
                ax.set_xticks([45,90])
  
            if density:
                ax.set_yticks([1e-4,1e-2,1e0])
            else:
                ax.set_yticks([1e0,1e2,1e4])
            
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    if W.shape[0]>1:
        fig.subplots_adjust(left=.1, bottom=.1, right=.95, top=.95,
                  wspace=0.05, hspace=0.05)
    else:
        fig.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
                  wspace=0.05, hspace=0.05)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_angles_broken_axis(W,W_0,costs,cmap=plt.cm.viridis,
                            savePath=None,density=True):
    col = np.linspace(0,1,W.shape[0])
    fig,(ax, ax2) = plt.subplots(1,2,sharey=True)
    fig.set_size_inches((4,4))

    for j in xrange(W.shape[0]):
        angles = dgcs.compute_angles(W[j])
        h,b = np.histogram(angles,np.arange(0,91)) 
        if density:
            h = h*1./np.sum(h)
        n= 45
        b = np.arange(1,91)
        ax.plot(b[:n],h[:n],drawstyle='steps-pre',
                color=cmap(col[j]),lw=1.5,label=costs[j])
        
        ax2.plot(b[n:],h[n:],drawstyle='steps-pre',
                color=cmap(col[j]),lw=1.5,label=costs[j])

        angles0 = dgcs.compute_angles(W_0)
        h0,b0 = np.histogram(angles0,np.arange(0,91))
        if density:
            h0 = h0*1./np.sum(h0)

        b0 = np.arange(1,91)
        ax.plot(b0[:n],h0[:n],'--',drawstyle='steps-pre',color='r',lw=1)
	ax2.plot(b0[n:],h0[n:],'--',drawstyle='steps-pre',color='r',lw=1)

        # hide the spines between ax and ax2
        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.yaxis.tick_right()
        ax2.tick_params(labelright='off')
        ax.yaxis.tick_left()
        
        ax.set_yscale('log')
        ax.set_xlim(0,90) 

        ax2.set_yscale('log')
        ax2.set_xlim(0,90) 
        
        if density:
            ax.set_ylim(1e-4,1e0)
        else:
            ax.set_ylim(1e0,1e4) 

        ax.legend(loc='upper left', frameon=False,fontsize=12,ncol=1)
        if density:
            ax.set_ylabel('Density',fontsize=16,labelpad=-2)
        else:
            ax.set_ylabel('Counts',fontsize=16)
        ax.set_xlim(0,11)
        ax.set_xticks([0,10])

        if density:
            ax.set_yticks([1e-4,1e-2,1e0])
        else:
            ax.set_yticks([1e0,1e2,1e4])

        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        #ax2.legend(loc='upper left', frameon=False,fontsize=12,ncol=1)
        ax2.set_xlim(80,90)
        ax2.set_xticks([81,90])
     
        if density:
            ax2.set_yticks([1e-4,1e-2,1e0])
        else:
            ax2.set_yticks([1e0,1e2,1e4])

        ax2.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((-d, + d), ( -d, + d), **kwargs)  # bottom-right diagonal
    
    fig.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
              wspace=0.05, hspace=0.05)

    fig.text(.525,.0125,r'$\theta$',fontsize=16)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_figure2b(W=None,W_0=None):
    overcompleteness = 2
    n_mixtures = 64
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['random','pathological']
    degeneracy_controls = ['QUASI-ORTHO','L2','COULOMB','RANDOM','L4']
    if W is None and W_0 is None:
        W, W_0 = dgcs.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                     n_sources, n_mixtures)
    costs = ['Quasi-ortho',r'$L_2$', 'Coulomb', 'Rand. prior', r'$L_4$']
    plot_angles_broken_axis(W[1],W_0[1],costs,density=True)


def plot_figure2a(W=None,W_0=None):
    overcompleteness = 2
    n_mixtures = 64
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['random','pathological']
    degeneracy_controls = ['QUASI-ORTHO','L2','COULOMB','RANDOM','L4']
    if W is None and W_0 is None:
        W, W_0 = dgcs.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                     n_sources, n_mixtures)
    costs = ['Quasi-ortho',r'$L_2$', 'Coulomb', 'Rand. prior', r'$L_4$']
    plot_angles_1column(W[0][np.newaxis,...],W_0,costs,density=True)


def plot_figure2c():
    plot_costs()

def plot_figure2d():
    plot_costs(mode=1)


def plot_filters(filters,n_pixels=8,n_filters=16,\
                 savePath=None,ax=None,figname='filters'):
    if ax is None:
        fig = plt.figure(figname)
        fig.clf()
        ax = plt.axes()
    im = tri(filters,(n_pixels,n_pixels),(n_filters,n_filters),
                       (2,2), scale_rows_to_unit_interval=False,
                       output_pixel_vals=False)
    ax.imshow(im,aspect='auto',interpolation='nearest',cmap='gray')
    ax.set_axis_off()
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_figure3a(angles,labels,density=True,\
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
    ax.set_xlim(20,90)
    ax.legend(loc='best', frameon=False,fontsize=12,ncol=1)
    ax.set_xlabel(r'$\theta$',fontsize=16,labelpad=0)
    ax.set_xticks([20,55,90])
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_figure3(bases=None,oc=4,lambd=10.,savePath=None,
                 costs = ['L2','COULOMB','RANDOM','L4']):
    #if not given, compute bases
    if bases is None:
        X, K = ds.generate_data(demo=1)[1:3]
        bases = learn_bases(X, costs=costs, oc=oc,lambd=lambd)
    #compute the angles
    n_sources = bases.shape[0]
    angles = np.zeros((len(costs),(n_sources**2-n_sources)/2))
    for i in xrange(len(costs)):
        angles[i] = dgcs.compute_angles(bases[i])
    #generate figure
    fig = plt.figure('Figure3',figsize=(6,3))
    fig.clf()
    labels = [r'$L_2$','Coulomb','Random prior',r'$L_4$']
    #figure3a
    ax_angles = plt.axes([.125,.15,.35,.7])
    plot_figure3a(angles,labels,density=True,ax=ax_angles)
    #figure3b
    ax_bases = plt.axes([.55,.15,.4,.8])
    w = bases[-1]
    plot_filters(w.dot(K),ax=ax_bases)
    fig.text(.01,.9,'a)',fontsize=14)
    fig.text(.5,.9,'b)',fontsize=14)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def learn_bases(X, costs=['L2','COULOMB','RANDOM','L4'],oc=4,lambd=10.):
    n_mixtures = X.shape[0]
    n_sources  = n_mixtures*oc
    bases = np.zeros((len(costs),n_sources,n_mixtures))
    for i in xrange(len(costs)):
        ica = ocica.ICA(n_mixtures=n_mixtures,n_sources=n_sources,lambd=lambd,
                        degeneracy=costs[i],optimizer='sgd',learning_rule=adam)
        ica.fit(X)
        bases[i] = ica.components_
    return bases

def get_Gabor_params(bases):
    if len(bases.shape)==2:
	bases = bases[np.newaxis,...]
    params = []
    for i in xrange(bases.shape[0]):
        fitter = fit.GaborFit()
        n_sources,n_mixtures = bases[i].shape
        l = np.sqrt(n_mixtures)
        w = bases[i].reshape((n_sources,l,l)).T
        params.append(fitter.fit(w))
    return params

def plot_GaborFit_xy(params,color=.5,savePath=None):
    color = plt.cm.viridis(color)
    fig = plt.figure('xy',figsize=(2,2))
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    freq = params[1][4]
    indices = np.arange(len(freq))#np.where(freq>1)[0]
    max_vx = np.max(params[1][5])*5
    max_vy = np.max(params[1][6])*5
    for i in indices:
        x = params[1][0][i]
        y = params[1][1][i]
        theta = params[1][2][i]/np.pi*180
        varx  = params[1][5][i]/max_vx
        vary  = params[1][6][i]/max_vy
        ax.add_patch(plt.Rectangle((x,y),width=varx,
                                   height=vary,angle=theta,
                                   facecolor=color,edgecolor=color,
                                   alpha=.8))
    ax.set_xlim(1,6)
    ax.set_ylim(1,6)
    ax.set_xlabel('X',fontsize=14)
    ax.set_ylabel('Y',fontsize=14)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_GaborFit_polar(params,color=.5,savePath=None):
    color = plt.cm.viridis(color)
    fig = plt.figure('polar',figsize=(2,2))
    fig.clf()
    ax = polar(fig)
    freq = params[1][4]/np.max(params[1][4])
    theta = params[1][2]/np.pi*180
    ax.plot(theta,freq,'.',color=color,ms=5,mew=1)
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()

def plot_GaborFit_envelope(params,color=.5,savePath=None):
    color = plt.cm.viridis(color)
    fig = plt.figure('polar',figsize=(2,2))
    fig.clf()
    ax = plt.axes([.15,.15,.8,.8])
    max_vx = np.max(params[1][5])*5
    max_vy = np.max(params[1][6])*5
    freq = params[1][4]/np.max(params[1][4])/200.
    indices = np.arange(len(freq))#np.where(freq>1)[0]
    for i in indices:
        varx  = params[1][5][i]/max_vx
        vary  = params[1][6][i]/max_vy
        ax.add_patch(plt.Circle((varx,vary),radius=freq[i],
                                   facecolor=color,edgecolor=color,
                                   alpha=.8))
    ax.set_xlim(.0,.2)
    ax.set_ylim(.0,.2)
    ax.set_xlabel(r'var[$\parallel$]',fontsize=14)
    ax.set_ylabel(r'var[$\perp$]',fontsize=14)
   
    if savePath is not None:
        plt.savefig(savePath,dpi=300)
    else:
        plt.show()



