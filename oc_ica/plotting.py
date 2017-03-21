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
import matplotlib.patheffects as pe


mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize']  = 14
mpl.rcParams['legend.fontsize'] = 10

from oc_ica import utils
#reload(utils)
from oc_ica.utils import tile_raster_images as tri
from oc_ica import analysis
reload(analysis)

import oc_ica.models.ica as ocica
#reload(ocica)
from oc_ica import datasets as ds
#reload(ds)
from oc_ica import gabor_fit as fit
#reload(fit)
from oc_ica import styles

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

    f, ax = plt.subplots(1,
                         sharex=True,
                         figsize=(6, 2))
    for ii, e in enumerate(l2_vals):
        if ii == 0:
            label = styles.labels['2']
        else:
            label = None
        ax.plot(thetas, e/9.,
                c=styles.colors['2'],
                lw=2, label=label,
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    for ii, e in enumerate(l4_vals):
        if ii == 0:
            label = styles.labels['4']
        else:
            label = None
        ax.plot(thetas, e/29., c=styles.colors['4'], lw=2,
                label=label, path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

    ax.grid()

    ax.set_xlim([thetas[0], thetas[-1]])
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.linspace(-1, 1, 2))
    ax.set_xticks(np.linspace(thetas[0], thetas[-1], 3))
    ax.set_xticklabels((np.linspace(thetas[0], thetas[-1], 3) /
                         np.pi*180.).astype(int))
    ax.set_xlabel(r'$\theta_2$')
    ax.set_ylabel(r'$e_i$ (arb. units)', labelpad=-0)
    ax.legend(loc='lower right', frameon=False)

    f.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return f, ax


def plot_figure2a(save_path=None, n_iter=10):
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
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['random']
    degeneracy_controls = ['QUASI-ORTHO', '2', 'COHERENCE_SOFT', 'COULOMB',
                           'RANDOM', '4']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    f, ax = plot_angles_1column(W, W_0, degeneracy_controls,
                                density=True)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return f, ax


def plot_figure2b(save_path=None, n_iter=10):
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
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['pathological']
    degeneracy_controls = ['QUASI-ORTHO', '2', 'COHERENCE_SOFT', 'COULOMB',
                           'RANDOM', '4']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    f, ax = plot_angles_broken_axis(W, W_0, degeneracy_controls,
                                    with_legend=False)
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()
    return f, ax


def plot_figure2de(panel, eps=1e-2,
                   legend=False, save_path=None):
    """
    Reproduces the panels c and d of figure 2.

    Parameters
    ----------
    panel: string, optional
         Which panel, options: 'd', 'e' 
    save_path: string, optional
         figure_path+figure_name+.format to store the figure. 
         If figure is stored, it is not displayed.   
    """
    formatter = mpl.ticker.StrMethodFormatter('{x:.2g}')
    fig = plt.figure('costs',figsize=(3,1.5))
    fig.clf()
    ax = plt.axes([.16,.15,.8,.81])
    costs = ['2', 'COULOMB', 'RANDOM', '4']
    col = np.linspace(0,1,len(costs))
    if panel=='d':
        xx = np.linspace(.6, 1., 100)
        fun = [lambda x: x,
               lambda x: x/(1.+eps-x**2)**(3/2),
               lambda x: (2*x)/(1.+eps-x**2),
               lambda x: x**3] 
        ax.set_yscale('log')
    elif panel=='e':
        xx = np.linspace(-.2, .2, 100)
        fun = [lambda x: x,
               lambda x: x/(1.+eps-x**2)**(3/2),
               lambda x: (2*x)/(1.+eps-x**2),
               lambda x: x**3] 
    else:
        raise ValueError('Choose c or d')

    for ii, cost in enumerate(costs):
        ax.plot(xx, fun[ii](xx), styles.line_styles[cost],
                c=styles.colors[cost], lw=2,
                label=styles.labels[cost],
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])

    if panel == 'd':
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    elif panel=='e':
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)

    ax.spines['left'].set_smart_bounds(True)
    ax.spines['bottom'].set_smart_bounds(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlim(np.min(xx),np.max(xx))
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_zorder(-10)

    if panel=='d':
        ax.set_ylim([1e-1, 5e1])
        ax.set_xticks(np.arange(0.6, 1.1, .2))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',labelpad=-4)#20)
        ax.set_xlabel(r'$\cos\,\theta$')
    elif panel=='e':
        ax.set_ylim(-.1,.1)
        ax.set_xticks(np.arange(-.2,.21,.2))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_yticks(np.arange(-.1,.11,.1))
        ax.yaxis.get_major_ticks()[1].set_visible(False)
        ax.xaxis.get_majorticklabels()[1].set_horizontalalignment('left')
        ax.yaxis.set_major_formatter(formatter)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',labelpad=70)#20)
        ax.set_xlabel(r'$\cos\,\theta$',labelpad=80)#180)
    else:
        raise ValueError
    if legend:
        ax.legend(loc='upper left',frameon=False,ncol=1)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return fig, ax


def plot_figure2c(save_path=None, n_iter=10):
    """
    Parameters:
    ----------
    save_path: string, optional
           Figure_path+figure_name+.format to store the figure. 
           If figure is stored, it is not displayed.   
    """
    rng = np.random.RandomState(20161206)
    overcompleteness = 2
    n_mixtures = 64
    n_sources = n_mixtures*overcompleteness
    initial_conditions = ['random']
    degeneracy_controls = ['COULOMB',
                           'RANDOM', 'COULOMB_F', 'RANDOM_F']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    f, ax = plot_angles_1column(W, W_0, degeneracy_controls,
                                plot_init=False, density=True,
                                pe_style=[True, True, False, False])
    ax.set_xlim(81.5, 90)
    ax.set_xticks([82, 90])
    ax.legend(loc='lower center', frameon=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return f, ax


def plot_angles_1column(W, W_0, costs, cmap=plt.cm.viridis,
                        density=True, plot_init=True,
                        pe_style=None):
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

    for ii, (ws, cost) in enumerate(zip(W, costs)):
        if cost[0] == 'L':
            cost = cost[1]
        elif cost == 'COHERENCE':
            cost = 'COHERENCE_SOFT'
        if ws.ndim  == 2:
            ws = ws[np.newaxis, ...]
        angles = np.array([])
        for wi in ws:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h, b = np.histogram(angles,np.arange(0,91)) 
        if density:
            h = h*1./np.sum(h)
        b = np.arange(1, 91)
        st = styles.line_styles[cost]
        c = styles.colors[cost]
        label = styles.labels[cost]
        pe_arg = [pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()]
        if pe_style is not None:
            if not pe_style[ii]:
                pe_arg = None
        ax.plot(b, h, st, drawstyle='steps-pre', color=c,
                lw=1.5, label=label,
                path_effects=pe_arg)

    if plot_init:
        if W_0.ndim == 2:
            W_0 = W_0[np.newaxis, ...]

        angles = np.array([])
        for wi in W_0:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h0, b0 = np.histogram(angles, np.arange(0,91))
        if density:
            h0 = h0 / np.sum(h0)
        b0 = np.arange(1, 91)
        ax.plot(b0, h0, styles.line_styles['INIT'], drawstyle='steps-pre',
                color=styles.colors['INIT'], lw=1)

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

    for ws, cost in zip(W, costs):
        if ws.ndim  == 2:
            ws = ws[np.newaxis, ...]
        angles = np.array([])
        for wi in ws:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h, b = np.histogram(angles,np.arange(0,91)) 
        if density:
            h = h*1./np.sum(h)
        b = np.arange(1, 91)
        c = styles.colors[cost]
        label = styles.labels[cost]
        st = styles.line_styles[cost]
        ax1.plot(b[:n], h[:n], st, drawstyle='steps-pre',
                 color=c, lw=1.5, label=label,
                 path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()])
    
        ax2.plot(b[n:], h[n:], st, drawstyle='steps-pre',
                 color=c, lw=1.5, label=label,
                 path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()])

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


def plot_bases(bases, fax=None, save_path=None, scale_rows=True):
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
    if fax is None:
        f, ax = plt.subplots(1)
    else:
        f, ax = fax
    im = tri(bases,(n_pixels,n_pixels),(n_bases,n_bases),
                (2,2), scale_rows_to_unit_interval=scale_rows,
                output_pixel_vals=False)
    ax.imshow(im,aspect='auto', interpolation='nearest', cmap='gray')
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    return f, ax


def plot_figure3ab(angles, models, xticks=None,
                   show_label=None,
                   density=True,
                   save_path=None, ax=None):
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
    if xticks is None:
        xticks = [0, 90]
    if show_label is None:
        show_label = [True] * len(models)
    show = False
    if ax is None:
        fig = plt.figure('angle_hist',figsize=(4,4))
        fig.clf()
        ax = plt.axes([.15,.1,.8,.8])
        show = True
    for ii, m in enumerate(models):
        h,b = np.histogram(angles[ii],np.arange(0,91))
        if density:
             h= h/np.sum(h)
        if show_label[ii]:
            label = styles.labels[m]
        else:
            label = None
        c = styles.colors[m]
        fmt = styles.line_styles[m]
        ax.plot(b[:-1], h, fmt, drawstyle='steps', color=c, lw=2, label=label,
                path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    ax.set_yscale('log')
    if not density:
        ax.set_ylabel('Counts')
        ax.set_yticks([1e0,1e2,1e4])
    else:
        ax.set_ylabel('Density',labelpad=-10)
        ax.set_yticks([1e-5,1e0])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(loc='upper left', frameon=False,ncol=1)
    ax.set_xlabel(r'$\theta$',labelpad=0)
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0], xticks[-1])


def plot_figure3(bases1, models1, bases_idx, bases2, models2, save_path=None):
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
    #compute the angles
    #generate figure
    f = plt.figure(figsize=(6, 8))
    ax_angles1 = plt.subplot2grid((3, 2), (0, 0))
    ax_angles2 = plt.subplot2grid((3, 2), (0, 1))
    ax_bases = plt.subplot2grid((3, 2), (1, 0), colspan=2, rowspan=2)

    #figure3a
    n_sources = bases1.shape[-2]
    n_iter = bases1.shape[1]
    angles = np.zeros((len(models1),n_iter * (n_sources**2-n_sources)/2))
    for ii, b in enumerate(bases1):
        angles[ii] = analysis.compute_angles(b)
    plot_figure3ab(angles, models1, density=True, ax=ax_angles1)

    #figure3b
    n_sources = bases2.shape[-2]
    n_iter = bases2.shape[1]
    angles = np.zeros((len(models2),n_iter * (n_sources**2-n_sources)/2))
    for ii, b in enumerate(bases2):
        angles[ii] = analysis.compute_angles(b)
    show_label = [False] * len(models2)
    show_label[-1] = True
    show_label[-3] = True
    plot_figure3ab(angles, models2, xticks=[45, 90], show_label=show_label,
                   density=True, ax=ax_angles2)

    #figure3b
    #ax_bases = plt.axes([.55,.15,.4,.8])
    w = bases1[bases_idx]
    if w.ndim == 3:
        w = w[0]
    plot_bases(w, fax=(f, ax_bases)) #, scale_rows=False)
    f.text(.01, .98, 'a)', fontsize=14)
    f.text(.5, .98, 'b)', fontsize=14)
    f.text(.01, .65, 'c)', fontsize=14)
    plt.tight_layout()
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
    vx = params[5]
    vy = params[6]
    ax = fractional_polar_axes(fig, rlim=(0, 1.5))
    for ii in range(len(freq)):
        ax.plot(theta[ii], freq[ii], 'o',
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=.6,
                ms=2. * np.sqrt(vx[ii] * vy[ii]), mew=1)
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
                 markerfacecolor=color,
                 markeredgecolor=color, alpha=.6)
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
