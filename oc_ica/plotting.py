from __future__ import division
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import angle_helper
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axisartist.floating_axes import GridHelperCurveLinear, FloatingSubplot


mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize']  = 14
mpl.rcParams['legend.fontsize'] = 10

from oc_ica import utils
reload(utils)
from oc_ica.utils import tile_raster_images as tri
from oc_ica import analysis
reload(analysis)

import oc_ica.models.ica as ocica
reload(ocica)
from oc_ica import datasets as ds
reload(ds)
from oc_ica import gabor_fit as fit
reload(fit)

model_color = {}


def plot_figure1c(save_path=None):
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    n_pts = 100
    l2_evals = [lambda th: th*0., 
                lambda th: 8*sin(th)**2,
                lambda th: 8*cos(th)**2 ]
    l4_evals = [lambda th: 4*(cos(2*th)-cos(4*th)),
                lambda th: -2*cos(2*th)-14*cos(4*th)
                           -sqrt(2)*sqrt(34-2*cos(2*th)+cos(4*th)-2*cos(6*th)+33*cos(8*th)),
                lambda th: -2*cos(2*th)-14*cos(4*th)
                           +sqrt(2)*sqrt(34-2*cos(2*th)+cos(4*th)-2*cos(6*th)+33*cos(8*th))]


    thetas = np.linspace(-np.pi/4., np.pi/4., n_pts)
    l2_vals = np.zeros((len(l2_evals), n_pts))
    l4_vals = np.zeros((len(l2_evals), n_pts))

    col = np.linspace(0, .75, len(l2_evals))

    for ii, f in enumerate(l2_evals):
        l2_vals[ii] = f(thetas)
    for ii, f in enumerate(l4_evals):
        l4_vals[ii] = f(thetas)

    f, (ax2, ax4) = plt.subplots(2, 1,
                                 sharex=True,
                                 figsize=(6, 3))
    for ii, e in enumerate(l2_vals):
        ax2.plot(thetas, e, label=r'$e_'+str(ii)+'$',
                 c=cm.plasma(col[ii]), lw=2)
    for ii, e in enumerate(l4_vals):
        ax4.plot(thetas, e, c=cm.plasma(col[ii]), lw=2)

    ax2.grid()
    ax4.grid()

    ax2.set_ylim([-.5, 8.5])
    ax2.set_yticks(np.linspace(0, 8, 2))
    leg = ax2.legend(loc='center right')
    ax2.set_ylabel(r'$L_2$ $e_i$', labelpad=14)

    ax4.set_xlim([thetas[0], thetas[-1]])
    ax4.set_yticks(np.linspace(-30, 30, 3))
    ax4.set_xticks(np.linspace(thetas[0], thetas[-1], 3))
    ax4.set_xticklabels((np.linspace(thetas[0], thetas[-1], 3) /
                         np.pi*180.).astype(int))
    ax4.set_xlabel(r'$\theta_2$')
    ax4.set_ylabel(r'$L_4$ $e_i$', labelpad=-0)

    f.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return f, (ax2, ax4)

def plot_figure2a(save_path=None):
    """Reproduces figure 2a of the NIPS16 paper
    Parameters:
    ----------
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """
    rng = np.random.RandomState(20161206)
    overcompleteness = 2
    n_mixtures = 64
    n_iter = 10
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['random']
    degeneracy_controls = ['QUASI-ORTHO', 'L2', 'COHERENCE_SOFT', 'COULOMB',
                           'RANDOM', 'L4']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p
    costs = ['Quasi-ortho',r'$L_2$', 'Soft Coherence', 'Coulomb',
             'Rand. prior', r'$L_4$']

    f, ax = plot_angles_1column(W, W_0, costs, density=True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return f, ax

def plot_figure2b(save_path=None):
    """Reproduces figure 2b of the NIPS16 paper.
    Parameters:
    ----------
    W     : array, optional
           Set of basis obtained by optimizing different costs.
           Dimension: n_costs X n_vectors X n_dims
    W_0   : array, optional
           Initial set of basis. 
           Dimension: n_costs X n_vectors X n_dims
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """
    rng = np.random.RandomState(20161206)
    overcompleteness = 2
    n_mixtures = 64
    n_iter = 10
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['pathological']
    degeneracy_controls = ['QUASI-ORTHO', 'L2', 'COHERENCE_SOFT', 'COULOMB',
                           'RANDOM', 'L4']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p
    costs = ['Quasi-ortho',r'$L_2$', 'Soft Coherence', 'Coulomb',
             'Rand. prior', r'$L_4$']
    f, ax = plot_angles_broken_axis(W,W_0,costs,density=True,with_legend=False)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()
    return f, ax


def plot_figure2cd(panel='c', eps=1e-2, save_path=None):
    """
    Reproduces the panels c and d of figure 2 in the NIPS16 paper

    Parameters
    ----------
    panel: string, optional
         Which panel, options: 'c', 'd' 
    save_path: string, optional
         figure_path+figure_name+.format to store the figure. 
         If figure is stored, it is not displayed.   
    """
    formatter = mpl.ticker.StrMethodFormatter('{x:.2g}')
    fig = plt.figure('costs',figsize=(3,3))
    fig.clf()
    ax = plt.axes([.16,.15,.8,.81])
    costs = [r'$L_2$', 'Coulomb', 'Random prior', r'$L_4$']
    col = np.linspace(0,1,len(costs))
    if panel=='c':
        xx = np.linspace(.6, 1., 100)
        fun = [lambda x: x,
               lambda x: x/(1.+eps-x**2)**(3/2),
               lambda x: (2*x)/(1.+eps-x**2),
               lambda x: x**3] 
        ax.set_yscale('log')
    elif panel=='d':
        xx = np.linspace(-.17, .17, 100)
        fun = [lambda x: x,
               lambda x: x/(1.+eps-x**2)**(3/2),
               lambda x: (2*x)/(1.+eps-x**2),
               lambda x: x**3] 

    for i in xrange(4):
        ax.plot(xx, fun[i](xx), color=cm.viridis(col[i]),
                lw=2, label=costs[i])

    if panel=='d':
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')

    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(np.min(xx),np.max(xx))
    ax.minorticks_off()

    if panel=='c':
        ax.legend(loc='upper left',frameon=False,ncol=1)
        ax.set_ylim([1e-1, 1e2])
        ax.set_xticks(np.arange(0.6, 1.1, .2))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',labelpad=-4)#20)
        ax.set_xlabel(r'$\cos\,\theta$')
    elif panel=='d':
        ax.set_ylim(-.1,.1)
        ax.set_xticks(np.arange(-.15,.16,.15))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_yticks(np.arange(-.1,.11,.05))
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',labelpad=65)#20)
        ax.set_xlabel(r'$\cos\,\theta$',labelpad=80)#180)
    else:
        raise ValueError

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    return fig

def plot_angles_1column(W, W_0, costs, cmap=plt.cm.viridis,
                        density=True):
    """
    Plots angle distributions of different costs and initial conditions.  
    Parameters:
    ----------
    W     : array
           Set of basis obtained by optimizing different costs.
           Dimension: (n_costs, n_vectors, n_dims)
           or
           Dimension: (n_costs, n_iter, n_vectors, n_dims)
    W_0   : array
           Initial set of basis.
           Dimension: (n_costs, n_vectors, n_dims)
           or
           Dimension: (n_costs, n_iter, n_vectors, n_dims)
    costs : list or array of strings
           Names of the costs that were evaluated
    cmap  : colormap object, optional
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    density: boolean, optional
           Use the density
    """
    n_costs = W.shape[0]
    col = np.linspace(0, 1, n_costs-1)
    col = np.hstack((np.zeros(1), col))
    figsize=(3, 3)
    f, ax = plt.subplots(1, figsize=figsize)

    for ii, ws in enumerate(W):
        if ws.ndim  == 2:
            ws = ws[np.newaxis, ...]
        angles = np.array([])
        for wi in ws:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h, b = np.histogram(angles,np.arange(0,91)) 
        if density:
            h = h*1./np.sum(h)
        b = np.arange(1, 91)
        if ii == 0:
            c = 'black'
        else:
            c = cmap(col[ii])
        ax.plot(b, h, drawstyle='steps-pre', color=c,
                lw=1.5, label=costs[ii])

    if W_0.ndim == 2:
        W_0 = W_0[np.newaxis, ...]

    angles = np.array([])
    for wi in W_0:
        angles = np.concatenate((analysis.compute_angles(wi), angles))
    h0, b0 = np.histogram(angles, np.arange(0,91))
    if density:
        h0 = h0 / np.sum(h0)
    b0 = np.arange(1, 91)
    ax.plot(b0, h0, '--', drawstyle='steps-pre', color='r', lw=1)
    ax.set_yscale('log')
    ax.set_xlim(0, 90) 
      
    if density:
        ax.set_ylim(1e-4, 1e0)
    else:
        ax.set_ylim(1e0, 1e4) 

    ax.legend(loc='upper left',frameon=False,ncol=1)
    ax.set_xlabel(r'$\theta$',labelpad=-10)
    if density:
        ax.set_ylabel('Density',labelpad=-2)
    else:
        ax.set_ylabel('Counts')
    ax.set_xlim(45, 90)
    ax.set_xticks([45, 90])

    if density:
        ax.set_yticks([1e-4, 1e-2, 1e0])
    else:
        ax.set_yticks([1e0, 1e2, 1e4])
    
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    if W.shape[0] > 1:
        f.subplots_adjust(left=.1, bottom=.1, right=.95, top=.95,
                          wspace=0.05, hspace=0.05)
    else:
        f.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
                          wspace=0.05, hspace=0.05)
    return f, ax

def plot_angles_broken_axis(W,W_0,costs,n=45, cmap=plt.cm.viridis,
                            save_path=None,density=True,with_legend=True):
    """Plots angle distributions of different costs and initial conditions  
    Parameters:
    ----------
    W     : array
           Set of basis obtained by optimizing different costs.
           Dimension: n_costs X n_vectors X n_dims
    W_0   : array
           Initial set of basis.
           Dimension: n_costs X n_vectors X n_dims
    costs : list or array of strings
           Names of the costs that are evaluated
    cmap  : colormap (plt.cm) object, optional
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    density: boolean, optional
           Use the density
    with_legend: boolean, optional
           Add a legend to the plot 
    """
    n_costs = W.shape[0]
    col = np.linspace(0, 1, n_costs-1)
    col = np.hstack((np.zeros(1), col))
    figsize=(3, 3)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ii, ws in enumerate(W):
        if ws.ndim  == 2:
            ws = ws[np.newaxis, ...]
        angles = np.array([])
        for wi in ws:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h, b = np.histogram(angles,np.arange(0,91)) 
        if density:
            h = h*1./np.sum(h)
        b = np.arange(1, 91)
        if ii == 0:
            c = 'black'
        else:
            c = cmap(col[ii])

        ax1.plot(b[:n],h[:n],drawstyle='steps-pre',
            color=c,lw=1.5,label=costs[ii])
    
        ax2.plot(b[n:],h[n:],drawstyle='steps-pre',
            color=c,lw=1.5,label=costs[ii])

    angles = np.array([])
    for wi in W_0:
        angles = np.concatenate((analysis.compute_angles(wi), angles))
    h0, b0 = np.histogram(angles, np.arange(0,91))
    if density:
        h0 = h0 / np.sum(h0)
    b0 = np.arange(1, 91)
    ax1.plot(b0[:n],h0[:n],'--',drawstyle='steps-pre',color='r',lw=1)
    ax2.plot(b0[n:],h0[n:],'--',drawstyle='steps-pre',color='r',lw=1)

    # hide the spines between ax and ax2
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright='off')
    ax1.yaxis.tick_left()
    
    ax1.set_yscale('log')
    ax1.set_xlim(0,90) 

    ax2.set_yscale('log')
    ax2.set_xlim(0,90) 
    
    if density:
        ax1.set_ylim(1e-4,1e0)
    else:
        ax1.set_ylim(1e0,1e4) 

    if with_legend:
        ax1.legend(loc='upper left', frameon=False,ncol=1)

    if density:
        ax1.set_ylabel('Density',labelpad=-2)
    else:
        ax1.set_ylabel('Counts')

    ax1.set_xlim(0,11)
    ax1.set_xticks([0,10])

    if density:
        ax1.set_yticks([1e-4,1e-2,1e0])
    else:
        ax1.set_yticks([1e0,1e2,1e4])

    ax1.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax2.set_xlim(79,90)
    ax2.set_xticks([80,90])
 
    if density:
        ax2.set_yticks([1e-4,1e-2,1e0])
    else:
        ax2.set_yticks([1e0,1e2,1e4])

    ax2.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((-d, + d), ( -d, + d), **kwargs)  # bottom-right diagonal
    
    f.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
              wspace=0.05, hspace=0.05)

    f.text(.525,.0125,r'$\theta$',fontsize=14)
    return f, (ax1, ax2)


def plot_bases(bases, save_path=None, scale_rows=True):
    """PLots a set of  bases. (Reproduces figure 3b of the NIPS16 paper.)
    Parameters:
    ----------
    bases : array
           Set of basis.
           Dimension: n_costs X n_vectors X n_dims
    ax    : Axes object, optional
           If None, the funtion generates a new Axes object.
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    figname: string, optional
           Name of the figure
    """
    n_pixels = int(np.sqrt(bases.shape[1]))
    n_bases  = int(np.sqrt(bases.shape[0]))
    f, ax = plt.subplots(1)
    im = tri(bases,(n_pixels,n_pixels),(n_bases,n_bases),
                (2,2), scale_rows_to_unit_interval=scale_rows,
                output_pixel_vals=False)
    ax.imshow(im,aspect='auto', interpolation='nearest', cmap='gray')
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()
    return f, ax

def plot_figure3a(angles,labels,density=True,\
                   save_path=None,ax=None):
    """Reproduces figure 3a of the NIPS16 paper
    Parameters:
    ----------
    angles: array
           Set of angles obtained by training different ICA models on natural images.
           Dimension: n_costs X n_angles
    ax    : Axes object, optional
           If None, the funtion generates a new Axes object.
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    figname: string, optional
           Name of the figure
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
        ax.set_ylabel('Counts')
        ax.set_yticks([1e0,1e2,1e4])
    else:
        ax.set_ylabel('Density',labelpad=-10)
        ax.set_yticks([1e-5,1e0])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(20,90)
    ax.legend(loc='best', frameon=False,ncol=1)
    ax.set_xlabel(r'$\theta$',labelpad=0)
    ax.set_xticks([20,55,90])
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()

def plot_figure3(bases=None,oc=2,lambd=10.,save_path=None):
    """Reproduces figure 3 of the NIPS16 paper
    Parameters:
    ----------
    bases: array
           Set of ICA bases. Dimension: n_costs X n_sources X n_mixtures
    oc    : int, optional
           Overcompleteness 
    lambd : float, optional
           Sparsity
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    figname: string, optional
           Name of the figure
    """
    costs = ['L2','COULOMB','RANDOM','L4']
    #if not given, compute bases
    if bases is None:
        X = ds.generate_data(demo_n=5)[1]
        print X.shape
        bases = learn_bases(X, costs=costs, oc=oc,lambd=lambd)
    #compute the angles
    n_sources = bases.shape[1]
    angles = np.zeros((len(costs),(n_sources**2-n_sources)/2))
    for i in xrange(len(costs)):
        angles[i] = analysis.compute_angles(bases[i])
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
    plot_bases(w,ax=ax_bases)
    fig.text(.01,.9,'a)',fontsize=14)
    fig.text(.5,.9,'b)',fontsize=14)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()


def fractional_polar_axes(f, thlim=(0, 180), rlim=(0, 1), step=(45, .5),
                          thlabel=r'$\theta$', rlabel='frequency', ticklabels=True):
    """
    Return polar axes that adhere to desired theta (in deg) and r limits. steps for theta
    and r are really just hints for the locators. Using negative values for rlim causes
    problems for GridHelperCurveLinear for some reason
    """
    th0, th1 = thlim # deg
    r0, r1 = rlim
    thstep, rstep = step

    # scale degrees to radians:
    tr_scale = Affine2D().scale(np.pi/180., 1.)
    tr = tr_scale + PolarAxes.PolarTransform()
    theta_grid_locator = angle_helper.LocatorDMS((th1-th0) // thstep)
    r_grid_locator = MaxNLocator((r1-r0) // rstep)
    theta_tick_formatter = angle_helper.FormatterDMS()
    grid_helper = GridHelperCurveLinear(tr,
                                        extremes=(th0, th1, r0, r1),
                                        grid_locator1=theta_grid_locator,
                                        grid_locator2=r_grid_locator,
                                        tick_formatter1=theta_tick_formatter,
                                        tick_formatter2=None)

    a = FloatingSubplot(f, 111, grid_helper=grid_helper)
    f.add_subplot(a)

    # adjust x axis (theta):
    a.axis["bottom"].set_visible(False)
    a.axis["top"].set_axis_direction("bottom") # tick direction
    a.axis["top"].toggle(ticklabels=ticklabels, label=bool(thlabel))
    a.axis["top"].major_ticklabels.set_axis_direction("top")
    a.axis["top"].label.set_axis_direction("top")

    # adjust y axis (r):
    a.axis["left"].set_axis_direction("bottom") # tick direction
    a.axis["right"].set_axis_direction("top") # tick direction
    a.axis["left"].toggle(ticklabels=ticklabels, label=bool(rlabel))

    # add labels:
    a.axis["top"].label.set_text(thlabel)
    a.axis["left"].label.set_text(rlabel)

    # create a parasite axes whose transData is theta, r:
    auxa = a.get_aux_axes(tr)
    # make aux_ax to have a clip path as in a?:
    auxa.patch = a.patch 
    # this has a side effect that the patch is drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to prevent this:
    a.patch.zorder = -2

    # add sector lines for both dimensions:
    thticks = grid_helper.grid_info['lon_info'][0]
    rticks = grid_helper.grid_info['lat_info'][0]
    for th in thticks[1:-1]: # all but the first and last
        auxa.plot([th, th], [r0, r1], '--', c='grey', zorder=-1)
    for ri, r in enumerate(rticks):
        # plot first r line as axes border in solid black only if it isn't at r=0
        if ri == 0 and r != 0:
            ls, lw, color = 'solid', 2, 'black'
        else:
            ls, lw, color = 'dashed', 1, 'grey'

        auxa.add_artist(plt.Circle([0, 0], radius=r, ls=ls, lw=lw, color=color, fill=False,
                        transform=auxa.transData._b, zorder=-1))
    return auxa

def get_Gabor_params(bases):
    """Fit Gabor funcions to a set of basis
    bases: array
           ICA bases. Dimension: n_costs X n_sources X n_mixtures
    """
    if bases.ndim == 2:
        bases = bases[np.newaxis,...]
    params = []
    fitter = fit.GaborFit()
    for ii, w  in enumerate(bases):
        n_sources, n_mixtures = w.shape
        l = int(np.around(np.sqrt(n_mixtures)))
        w = w.reshape((n_sources, l, l)).T
        params.append(fitter.fit(w))
    return params

def plot_GaborFit_xy(params, n_pixels, color=.5,
                     save_path=None,
                     ax=None, figsize=None):
    """
    Plot Gabor parameters using a "confetti plot": 
       - position of rectangle:  position of Gabor
       - width,height of rectangle:  variance of envelope 
       - angle of the rectangle: orientation of the Gabor
    Parameters:
    ----------
    params : list of arrays 
           Gabor parameters :x, y, orientation, phase, 
                frequency, varx, vary 
           Dimension of arrays : n_sources
    color : float, optional
           Color value for viridis colormap
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """ 
    color = plt.cm.viridis(color)
    if figsize is None:
        figsize = (2,2)
    if ax is None:
        fig = plt.figure('xy', figsize=figsize)   
        fig.clf()
        ax = plt.axes([.15,.15,.8,.8])
    max_vx = np.max(params[5])
    max_vy = np.max(params[6])
    xs = params[0]
    ys = params[1]
    for ii in range(params[0].size):
        x = xs[ii]
        y = ys[ii]
        theta = params[2][ii]/np.pi*180
        varx  = params[5][ii]/max_vx
        vary  = params[6][ii]/max_vy
        ax.add_patch(plt.Rectangle((x,y),
                                   width=varx,
                                   height=vary,
                                   angle=theta,
                                   facecolor=color,
                                   edgecolor=color,
                                   alpha=.6))
    ax.set_xlim(1, n_pixels-2)
    ax.set_ylim(1, n_pixels-2)
    ax.set_xlabel('x-position', fontsize=14)
    ax.set_ylabel('y-position', fontsize=14)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        pass#plt.show()

def plot_GaborFit_polar(params,color=.5,save_path=None, 
                        figsize=None):
    """Plot Gabor parameters using a polar plot: 
       - radius: frequency 
       - angle : orientation of the Gabor
    Parameters:
    ----------
    params : list of arrays 
           Gabor parameters :x, y, orientation, phase, 
                frequency, varx, vary 
           Dimension of arrays : n_sources
    color : float, optional
           Color value for viridis colormap
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """ 
    color = plt.cm.viridis(color)
    if figsize is None:
        figsize = (2,2)
    fig = plt.figure('polar',figsize=figsize)
    fig.clf()
    freq = params[4] / (2. * np.pi)
    theta = params[2]/np.pi*180 % 180
    max_vx = np.max(params[5])/50.
    max_vy = np.max(params[6])/50.
    ax = fractional_polar_axes(fig, rlim=(0, 1.5))
    ii = freq.argmax()
    print(freq[ii], theta[ii])
    ax.plot(theta, freq, 'o',
            markerfacecolor=color,
            markeredgecolor=color,
            alpha=.6,
            ms=5, mew=1)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        pass#plt.show()

def plot_GaborFit_envelope(params, color=.5, save_path=None,
                           ax=None, figsize=None):
    """Plot Gabor parameters using a scatter plot: 
       - position of circles: size of the evelope (varx,vary) 
       - size of the circle : frequency of the cosine function
    Parameters:
    ----------
    params : list of arrays 
           Gabor parameters :x, y, orientation, phase, 
                frequency, varx, vary 
           Dimension of arrays : n_sources
    color : float, optional
           Color value for viridis colormap
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """ 
    color = plt.cm.viridis(color)
    if figsize is None:
        figsize = (2,2)
    if ax is None:
        fig = plt.figure('envelope',figsize=figsize)
        fig.clf()
        ax = plt.axes([.15,.15,.8,.8])
    varxs  = params[5]
    varys  = params[6]
    freq = 5. * params[4] / (2. * np.pi)
    for ii in range(len(varxs)):
        varx  = varxs[ii]
        vary  = varys[ii]
        plt.plot(varx, vary, 'o', ms=freq[ii], mew=1,
                 color=color, alpha=.6)
    ax.set_xlim(1e-1,1e1)
    ax.set_ylim(1e-1,1e1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'var[$\parallel$]', fontsize=14)
    ax.set_ylabel(r'var[$\perp$]', fontsize=14)
    ax.minorticks_off()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        pass#plt.show()
