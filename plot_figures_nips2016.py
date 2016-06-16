from __future__ import division
import pdb
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from utils import tile_raster_images
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

import evaluate_degeneracy_controls as dgcs
reload(dgcs)

def plot_costs(mode=0,savePath=None):
    formatter=mpl.ticker.FormatStrFormatter('%.1f')
    fig = plt.figure('costs',figsize=(3,3))
    fig.clf()
    ax = plt.axes([.16,.15,.8,.81])
    costs = ['L2', 'COULOMB', 'RANDOM', 'L4']
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
        ax.plot(xx,fun[i](xx),color=cm.jet(col[i]),lw=2,label=costs[i])
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


def plot_angles_1column(W,W_0,costs,cmap=plt.cm.jet,
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
                #ax.set_xticks([0,45,90])
  
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



def plot_angles_broken_axis(W,W_0,costs,cmap=plt.cm.jet,
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
        ax2.tick_params(labelright='off')  # don't put tick labels at the top
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
        #ax.set_xlabel(r'$\theta$',fontsize=16,labelpad=-10)
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
        #ax2.set_xlabel(r'$\theta$',fontsize=16,labelpad=-10)
        #ax2.set_xlim(79,90)
        #ax2.set_xticks([80,90])
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


