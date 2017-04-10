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
from imp import reload



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
from oc_ica import styles
reload(styles)


def plot_figure1c(save_path=None, fax=None):
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

    if fax is None:
        f, ax = plt.subplots(1,
                             figsize=(5, 2))
    else:
        f, ax = fax
    for ii, e in enumerate(l2_vals):
        if ii == 0:
            label = styles.labels['2']
        else:
            label = None
        ax.plot(thetas, e/9.,
                c=styles.colors['2'],
                lw=styles.lw, label=label,
                path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
    for ii, e in enumerate(l4_vals):
        if ii == 0:
            label = styles.labels['4']
        else:
            label = None
        ax.plot(thetas, e/29., c=styles.colors['4'], lw=styles.lw,
                label=label, path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])

    ax.grid()

    ax.set_xlim([thetas[0], thetas[-1]])
    ax.set_ylim(-1, 1)
    ax.set_yticks(np.linspace(-1, 1, 3))
    ax.set_xticks(np.linspace(thetas[0], thetas[-1], 3))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_xticklabels((np.linspace(thetas[0], thetas[-1], 3) /
                         np.pi*180.).astype(int))
    ax.set_xlabel(r'$\theta_2$', fontsize=styles.label_fontsize)
    ax.set_ylabel(r'$e_i$ (arb. units)', labelpad=-0, fontsize=styles.label_fontsize)
    ax.legend(loc='lower right', frameon=False, fontsize=styles.legend_fontsize)
    ax.tick_params(labelsize=styles.ticklabel_fontsize)

    if fax is None:
        f.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

def plot_figure1(save_path=None):
    def add_arrow(ax, dx, dy, c):
        return ax.add_patch(mpl.patches.FancyArrow(0, 0, dx, dy, width=.02,
                                                length_includes_head=True, head_width=.1,
                                                head_length=.1,
                                                facecolor=c, edgecolor=c))
    def add_arrow(ax, dx, dy, c):
        return ax.quiver(0, 0, dx, dy,angles='xy', scale_units='xy', scale=1, width=.02,
                                                facecolor=c, edgecolor=c,
                        headaxislength=3, headlength=3)

    def add_arc(ax, r, theta1, theta2, c):
        return ax.add_patch(mpl.patches.Arc((0, 0), r, r, theta1=theta1, theta2=theta2,
                                              facecolor='none', edgecolor=c,
                                              lw=2*styles.lw))
    dtheta1 = .2
    dtheta2 = np.pi/4
    figsize = (4, 2.5)
    xlim = [-np.cos(dtheta2)-.05, 1.05]
    ylim = [-.05, 1.05]
    ratio = (np.diff(xlim)/np.diff(ylim))[0]

    top_lr_edge = .01
    bot_l_edge = .1
    bot_r_edge = .025
    top_edge = top_lr_edge
    bot_edge = .15
    top_gap = .001
    mid_gap = .04

    top_lr_edge_x = top_lr_edge * figsize[0]
    bot_l_edge_x = bot_l_edge * figsize[0]
    bot_r_edge_x = bot_r_edge * figsize[0]
    top_edge_y = top_edge * figsize[1]
    bot_edge_y = bot_edge * figsize[1]
    top_gap_x = top_gap * figsize[0]
    mid_gap_y = mid_gap * figsize[1]

    width_top = figsize[0]/2-top_lr_edge_x-top_gap_x/2
    width_bot = figsize[0]-bot_l_edge_x-bot_r_edge_x
    height_top = width_top / ratio

    width_top = width_top / figsize[0]
    height_top = height_top / figsize[1]
    width_bot = width_bot / figsize[0]

    f = plt.figure(figsize=figsize)
    ax1 = f.add_axes((top_lr_edge, 1-height_top+mid_gap/2, width_top, height_top))
    ax2 = f.add_axes((.5+top_gap/2, 1-height_top+mid_gap/2, width_top, height_top))
    ax3 = f.add_axes((bot_l_edge, bot_edge,
                      1.-bot_l_edge-bot_r_edge,
                      1-bot_edge-mid_gap-top_edge-height_top))

    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_axis_off()


    add_arrow(ax1, 1, 0, 'k')
    add_arrow(ax1, 0, 1, 'k')
    add_arrow(ax1, np.cos(dtheta1), np.sin(dtheta1), 'b')
    add_arrow(ax1, np.cos(np.pi/2+dtheta1), np.sin(np.pi/2+dtheta1), 'b')
    a = add_arc(ax1, .7, 0, dtheta1*180/np.pi, 'r')
    a.set_zorder(0)
    a = add_arc(ax1, .3, 0, 90, 'k')
    a.set_zorder(0)
    add_arc(ax1, .5, dtheta1*180/np.pi, 90+dtheta1*180/np.pi, 'b')
    ax1.text(.1, -.15, r'$\theta_1$', fontsize=styles.label_fontsize, color='k')
    ax1.text(.05, .3, r'$\theta_3$', fontsize=styles.label_fontsize, color='b')
    ax1.text(.3, -.15, r'$\theta_2$', fontsize=styles.label_fontsize, color='r')
    ax1.add_patch(mpl.patches.Circle((0, 0), .01, color='b'))

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_axis_off()


    add_arrow(ax2, 1, 0, 'k')
    add_arrow(ax2, 0, 1, 'k')
    add_arrow(ax2, np.cos(dtheta2), np.sin(dtheta2), 'b')
    add_arrow(ax2, np.cos(np.pi/2+dtheta2), np.sin(np.pi/2+dtheta2), 'b')
    a = add_arc(ax2, .7, 0, dtheta2*180/np.pi, 'r')
    a.set_zorder(0)
    a = add_arc(ax2, .3, 0, 90, 'k')
    a.set_zorder(0)
    add_arc(ax2, .5, dtheta2*180/np.pi, 90+dtheta2*180/np.pi, 'b')
    ax2.text(.1, -.15, r'$\theta_1$', fontsize=styles.label_fontsize, color='k')
    ax2.text(.05, .3, r'$\theta_3$', fontsize=styles.label_fontsize, color='b')
    ax2.text(.3, -.15, r'$\theta_2$', fontsize=styles.label_fontsize, color='r')
    ax2.add_patch(mpl.patches.Circle((0, 0), .01, color='b'))

    plot_figure1c(fax=(f, ax3))

    f.text(.00, .93, 'a)', fontsize=styles.letter_fontsize)
    f.text(.5, .93, 'b)', fontsize=styles.letter_fontsize)
    f.text(.00, .5, 'c)', fontsize=styles.letter_fontsize)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_figure2a(save_path=None, n_iter=10, faxes=None):
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
    degeneracy_controls = ['QUASI-ORTHO', '2', '4', 'COHERENCE_SOFT', 'COULOMB',
                           'RANDOM']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    faxes = plot_angles_broken_axis(W, W_0, degeneracy_controls, faxes=faxes)
    for ax in faxes[1]:
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
    faxes[1][0].yaxis.set_ticks_position('left')
    if save_path is not None:
        plt.savefig(save_path)


def plot_figure2b(save_path=None, n_iter=10, ax=None,
                  add_ylabel=False):
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
    degeneracy_controls = ['QUASI-ORTHO', '2', '4', 'COHERENCE_SOFT',
                           'RANDOM', 'COULOMB']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    ax = plot_angles_1column(W, W_0, degeneracy_controls,
                             ax=ax, add_ylabel=add_ylabel)
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if save_path is not None:
        plt.savefig(save_path)


def plot_figure2c(save_path=None, n_iter=10, ax=None):
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
    degeneracy_controls = ['COULOMB', 'RANDOM', 'COULOMB_F', 'RANDOM_F']
    W = np.full((len(degeneracy_controls), n_iter, n_sources,
                 n_mixtures), np.nan)
    W_0 = np.full((n_iter, n_sources, n_mixtures), np.nan)
    for ii in range(n_iter):
        Wp, W_0p = analysis.evaluate_dgcs(initial_conditions, degeneracy_controls,
                                          n_sources, n_mixtures, rng=rng)
        W[:, ii] = Wp[0]
        W_0[ii] = W_0p

    ax = plot_angles_1column(W, W_0, degeneracy_controls,
                        plot_init=False,
                        pe_style=[True, True, False, False],
                        ax=ax)
    ax.set_xlim(81.5, 90)
    ax.set_xticks([82, 90])
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.legend(loc='lower center', frameon=False,
            fontsize=styles.legend_fontsize)
    ax.tick_params(labelsize=styles.ticklabel_fontsize)
    if save_path is not None:
        plt.savefig(save_path)


def plot_figure2de(panel, eps=1e-2,
                   legend=False, save_path=None,
                   ax=None, add_xlabel=False):
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
    formatter = mpl.ticker.StrMethodFormatter('{x:.1g}')
    if ax is None:
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
                c=styles.colors[cost], lw=styles.lw,
                label=styles.labels[cost],
                path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])

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
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.set_xlim(np.min(xx),np.max(xx))
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set_zorder(-10)

    if panel=='d':
        ax.set_ylim([1e-1, 5e1])
        ax.set_xticks(np.arange(0.6, 1.1, .2))
        ax.xaxis.set_major_formatter(formatter)
        if add_xlabel:
            ax.set_xlabel(r'$\cos\,\theta$', fontsize=styles.label_fontsize)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',
                fontsize=styles.label_fontsize,
                labelpad=0)
        ax.tick_params(pad=0)
    elif panel=='e':
        ax.set_ylim(-.1,.1)
        ax.set_xticks(np.arange(-.2,.21,.2))
        ax.xaxis.set_major_formatter(formatter)
        ax.set_yticks(np.arange(-.1,.11,.1))
        ax.yaxis.get_major_ticks()[1].set_visible(False)
        ax.xaxis.get_majorticklabels()[1].set_horizontalalignment('left')
        ax.yaxis.set_major_formatter(formatter)
        if add_xlabel:
            ax.set_xlabel(r'$\cos\,\theta$', fontsize=styles.label_fontsize,
                          labelpad=25)
        ax.set_ylabel(r'$\nabla C(\cos\,\theta)$',
                fontsize=styles.label_fontsize,
                labelpad=60)
    else:
        raise ValueError
    if legend:
        ax.legend(loc='upper left',frameon=False,ncol=1,
                  fontsize=styles.legend_fontsize)

    if save_path is not None:
        plt.savefig(save_path)


def plot_figure2(save_path=None, n_iter=10):
    f = plt.figure(figsize=(5, 4.5))
    left_gap = .0825
    right_gap = .025
    bottom_gap = .04
    top_gap = .019
    gap = .0125
    width = (.5 - left_gap - right_gap - gap) / 2
    height = .5 - bottom_gap - top_gap
    ax1 = f.add_axes([left_gap, .5+bottom_gap, width, height])
    ax2 = f.add_axes([left_gap+width+gap, .5+bottom_gap, width, height])
    ax3 = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((4, 4), (2, 2), rowspan=1, colspan=2)
    ax6 = plt.subplot2grid((4, 4), (3, 2), rowspan=1, colspan=2)
    ax1.set_zorder(ax2.get_zorder()+1)
    plot_figure2a(n_iter=n_iter, faxes=(f, (ax1, ax2)))
    plot_figure2b(n_iter=n_iter, ax=ax3, add_ylabel=False)
    plot_figure2c(n_iter=n_iter, ax=ax4)
    plot_figure2de('d', ax=ax5)
    plot_figure2de('e', add_xlabel=True, ax=ax6)
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(labelsize=styles.ticklabel_fontsize)
        ax.tick_params(pad=-2)
    x1 = .0
    x2 = .5
    y1 = .945
    y2 = .46
    y3 = .24
    f.text(x1, y1, 'a)', fontsize=styles.letter_fontsize)
    f.text(x2, y1, 'b)', fontsize=styles.letter_fontsize)
    f.text(x1, y2, 'c)', fontsize=styles.letter_fontsize)
    f.text(x2, y2, 'd)', fontsize=styles.letter_fontsize)
    f.text(x2, y3, 'e)', fontsize=styles.letter_fontsize)
    f.tight_layout(pad=0, w_pad=0, h_pad=0)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_angles_1column(W, W_0, costs, cmap=plt.cm.viridis,
                        plot_init=True,
                        pe_style=None, legend=False,
                        ax=None, add_ylabel=True):
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
    """
    n_costs = W.shape[0]
    col = np.linspace(0, 1, n_costs-1)
    col = np.hstack((np.zeros(1), col))
    if ax is None:
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
        h, b = np.histogram(angles, styles.angle_bins) 
        b = b[1:]
        h = h*1./np.sum(h)
        st = styles.line_styles[cost]
        c = styles.colors[cost]
        label = styles.labels[cost]
        pe_arg = [pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()]
        if pe_style is not None:
            if not pe_style[ii]:
                pe_arg = None
        ax.plot(b, h, st, drawstyle=styles.ds, color=c,
                lw=styles.lw, label=label,
                path_effects=pe_arg)

    if plot_init:
        if W_0.ndim == 2:
            W_0 = W_0[np.newaxis, ...]

        angles = np.array([])
        for wi in W_0:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h0, b0 = np.histogram(angles, styles.angle_bins)
        b0 = b0[1:]
        h0 = h0 / np.sum(h0)
        ax.plot(b0, h0, styles.line_styles['INIT'], drawstyle=styles.ds,
                color=styles.colors['INIT'], lw=styles.lw)

    ax.set_yscale('log')
    ax.set_xlim(0, 90) 
      
    ax.set_ylim(1e-4, 1e0)
    if legend:
        ax.legend(loc='upper left',frameon=False,ncol=1,
                fontsize=styles.legend_fontsize)
    ax.set_xlabel(r'$\theta$',labelpad=-10, fontsize=styles.label_fontsize)
    if add_ylabel:
        ax.set_ylabel('Probability',labelpad=-2, fontsize=styles.label_fontsize)
    ax.set_xlim(45, 90)
    ax.set_xticks([45, 90])

    ax.set_yticks([1e-4, 1e-2, 1e0])
    
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    """
    if W.shape[0] > 1:
        f.subplots_adjust(left=.1, bottom=.1, right=.95, top=.95,
                          wspace=0.05, hspace=0.05)
    else:
        f.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
                          wspace=0.05, hspace=0.05)
    """
    return ax


def plot_angles_broken_axis(W,W_0,costs, cmap=plt.cm.viridis,
                            save_path=None, legend=True,
                            faxes=None):
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
    legend: boolean, optional
           Add a legend to the plot 
    """

    if faxes is None:
        figsize=(3, 3)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        f, (ax1, ax2) = faxes

    for ws, cost in zip(W, costs):
        if ws.ndim  == 2:
            ws = ws[np.newaxis, ...]
        angles = np.array([])
        for wi in ws:
            angles = np.concatenate((analysis.compute_angles(wi), angles))
        h, b = np.histogram(angles, styles.angle_bins) 
        b = b[1:]
        h = h*1./np.sum(h)
        c = styles.colors[cost]
        label = styles.labels[cost]
        st = styles.line_styles[cost]
        ax1.plot(b, h, st, drawstyle=styles.ds,
                 color=c, lw=styles.lw, label=label,
                 path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
        ax2.plot(b, h, st, drawstyle=styles.ds,
                 color=c, lw=styles.lw, label=label,
                 path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])

    angles = np.array([])
    for wi in W_0:
        angles = np.concatenate((analysis.compute_angles(wi), angles))
    h0, b0 = np.histogram(angles, styles.angle_bins)
    b0 = b0[1:]
    h0 = h0 / np.sum(h0)
    ax1.plot(b0, h0, styles.line_styles['INIT'], drawstyle=styles.ds,
             color=styles.colors['INIT'], lw=styles.lw)
    ax2.plot(b0,h0, styles.line_styles['INIT'], drawstyle=styles.ds,
             color=styles.colors['INIT'], lw=styles.lw)

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
    
    ax1.set_ylim(1e-4,1e0)

    if legend:
        ax1.legend(loc='upper left', frameon=False,ncol=1,
                fontsize=styles.legend_fontsize)

    ax1.set_ylabel('Probability',labelpad=-2, fontsize=styles.label_fontsize)

    ax1.set_xlim(0,11)
    ax1.set_xticks([0,10])

    ax1.set_yticks([1e-4,1e-2,1e0])

    ax1.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax2.set_xlim(79,90)
    ax2.set_xticks([80,90])
 
    ax2.set_yticks([1e-4,1e-2,1e0])

    ax2.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((-d, + d), ( -d, + d), **kwargs)  # bottom-right diagonal
    """
    f.subplots_adjust(left=.15, bottom=.1, right=.95, top=.95,
              wspace=0.05, hspace=0.05)

    f.text(.525,.0125,r'$\theta$')
    """
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
    n_bases  = bases.shape[0]
    if fax is None:
        f, ax = plt.subplots(1)
    else:
        f, ax = fax
    im = tri(bases,(n_pixels,n_pixels),(8,n_bases//8),
                (2,2), scale_rows_to_unit_interval=scale_rows,
                output_pixel_vals=False)
    ax.imshow(im,aspect='auto', interpolation='nearest', cmap='gray')
    ax.set_axis_off()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    return f, ax


def plot_figure3ab(angles, models, xticks=None,
                   show_label=None,
                   save_path=None, ax=None,
                   add_ylabel=False):
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
        h, b = np.histogram(angles[ii], styles.angle_bins)
        h= h/np.sum(h)
        if show_label[ii]:
            label = styles.labels[m]
        else:
            label = None
        c = styles.colors[m]
        fmt = styles.line_styles[m]
        ax.plot(b[1:], h, fmt, drawstyle=styles.ds, color=c, lw=styles.lw, label=label,
                path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
    ax.set_yscale('log')
    if add_ylabel:
        ax.set_ylabel('Probability',labelpad=-10, fontsize=styles.label_fontsize)
    ax.set_yticks([1e-5,1e0])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.legend(loc='upper left', frameon=False,ncol=1,
            fontsize=styles.legend_fontsize)
    ax.set_xlabel(r'$\theta$',labelpad=0, fontsize=styles.label_fontsize)
    ax.set_xticks(xticks)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.tick_params(labelsize=styles.ticklabel_fontsize)
    ax.tick_params(pad=0)


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
    f = plt.figure(figsize=(4, 4))
    ax_angles1 = plt.subplot2grid((2, 2), (0, 0))
    ax_angles1.get_yaxis().set_tick_params(direction='out')
    ax_angles1.get_xaxis().set_tick_params(direction='out')
    ax_angles1.xaxis.set_ticks_position('bottom')
    ax_angles1.yaxis.set_ticks_position('left')

    ax_angles2 = plt.subplot2grid((2, 2), (0, 1))
    ax_angles2.get_yaxis().set_tick_params(direction='out')
    ax_angles2.get_xaxis().set_tick_params(direction='out')
    ax_angles2.xaxis.set_ticks_position('bottom')
    ax_angles2.yaxis.set_ticks_position('left')

    ax_bases = plt.subplot2grid((2, 2), (1, 0), colspan=2, rowspan=1)

    #figure3a
    n_sources = bases1.shape[-2]
    n_iter = bases1.shape[1]
    angles = np.zeros((len(models1),n_iter *
                       int(np.around((n_sources**2-n_sources)/2.))))
    for ii, b in enumerate(bases1):
        angles[ii] = analysis.compute_angles(b)
    plot_figure3ab(angles, models1, ax=ax_angles1,
                   add_ylabel=True)

    #figure3b
    n_sources = bases2.shape[-2]
    n_iter = bases2.shape[1]
    angles = np.zeros((len(models1),n_iter *
                       int(np.around((n_sources**2-n_sources)/2.))))
    for ii, b in enumerate(bases2):
        angles[ii] = analysis.compute_angles(b)
    show_label = [False] * len(models2)
    show_label[-1] = True
    show_label[-3] = True
    plot_figure3ab(angles, models2, xticks=[45, 90], show_label=show_label,
                   ax=ax_angles2)

    #figure3b
    #ax_bases = plt.axes([.55,.15,.4,.8])
    w = bases1[bases_idx]
    if w.ndim == 3:
        w = w[0]
    plot_bases(w, fax=(f, ax_bases)) #, scale_rows=False)
    f.text(.01, .89, 'a)', fontsize=styles.letter_fontsize)
    f.text(.49, .89, 'b)', fontsize=styles.letter_fontsize)
    f.text(.01, .45, 'c)', fontsize=styles.letter_fontsize)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
    else:
        plt.show()


def fractional_polar_axes(f, thlim=(0, 180), rlim=(0, 1), step=(45, .75),
                          thlabel=r'$\phi$', rlabel='frequency',
                          ticklabels=True,
                          pos=None, labelx=True, labely=True):
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
    grid_helper = GridHelperCurveLinear(tr,
                                        extremes=(th0, th1, r0, r1),
                                        grid_locator1=theta_grid_locator,
                                        grid_locator2=r_grid_locator)

    a = FloatingSubplot(f, *pos, grid_helper=grid_helper)
    f.add_subplot(a)

    # adjust x axis (theta):
    a.axis["bottom"].set_visible(False)
    a.axis["top"].set_axis_direction("bottom") # tick direction
    a.axis["top"].toggle(ticklabels=ticklabels, label=bool(thlabel))
    a.axis["top"].major_ticklabels.set_axis_direction("top")
    a.axis["top"].label.set_axis_direction("top")
    a.axis['top'].major_ticklabels.set_pad(1)
    a.axis['top'].major_ticklabels.set_size(styles.label_fontsize)

    # adjust y axis (r):
    a.axis["left"].set_axis_direction("bottom") # tick direction
    a.axis["right"].set_axis_direction("top") # tick direction
    a.axis["left"].toggle(ticklabels=ticklabels, label=bool(rlabel))
    a.axis['left'].major_ticklabels.set_size(styles.label_fontsize)

    # add labels:
    if labely:
        a.axis["top"].label.set_text(thlabel)
        a.axis['top'].label.set_size(styles.label_fontsize)
    if labelx:
        a.axis["left"].label.set_text(rlabel)
        a.axis['left'].label.set_size(styles.label_fontsize)

    # create a parasite axes whose transData is theta, r:
    auxa = a.get_aux_axes(tr)
    # make aux_ax to have a clip path as in a?:
    auxa.patch = a.patch 
    # this has a side effect that the patch is drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to prevent this:
    a.patch.zorder = -2
    a.tick_params(labelsize=styles.ticklabel_fontsize)
    a.tick_params(pad=0)
    auxa.tick_params(labelsize=styles.ticklabel_fontsize)
    auxa.tick_params(pad=0)

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


def plot_GaborFit_xy(params, n_pixels, model,
                     save_path=None,
                     ax=None, figsize=None,
                     f=None, pos=None,
                     labelx=True, labely=True):
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
    color = styles.colors[model]
    if figsize is None:
        figsize = (2,2)
    if ax is None and f is None:
        fig = plt.figure('xy', figsize=figsize)   
        fig.clf()
        ax = plt.axes([.15,.15,.8,.8])
    elif ax is None:
        ax = f.add_subplot(*pos)

    max_vx = np.max(params[5])
    max_vy = np.max(params[6])
    xs = params[0]
    ys = params[1]
    """
    print xs.min(), xs.max()
    print ys.min(), xs.max()
    print np.sqrt(np.exp(params[5]))#/max_vx
    print np.sqrt(np.exp(params[6]))#/max_vy
    """
    for ii in range(params[0].size):
        x = xs[ii]
        y = ys[ii]
        theta = params[2][ii]/np.pi*180
        varx  = np.exp(params[5][ii])#/max_vx
        vary  = np.exp(params[6][ii])#/max_vy
        ax.add_patch(plt.Rectangle((x,y),
                                   width=np.sqrt(varx)/5.,
                                   height=np.sqrt(vary)/5.,
                                   angle=theta,
                                   facecolor=color,
                                   edgecolor='k',
                                   alpha=.4))
    ax.set_xlim(1, n_pixels-2)
    ax.set_ylim(1, n_pixels-2)
    if labelx:
        ax.set_xlabel('x-position', labelpad=-1, fontsize=styles.label_fontsize)
    else:
        ax.set_xticklabels([])
    if labely:
        ax.set_ylabel('y-position', fontsize=styles.label_fontsize)
    else:
        ax.set_yticklabels([])
    ax.set_title(styles.labels[model], fontsize=styles.label_fontsize)
    pass
    pass

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.tick_params(labelsize=styles.ticklabel_fontsize)
    ax.tick_params(pad=0)
    if save_path is not None:
        plt.savefig(save_path)


def plot_GaborFit_polar(params, model, save_path=None, 
                        figsize=None, f=None, pos=None,
                        labelx=True, labely=True):
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
    color = styles.colors[model]
    if f is None:
        if figsize is None:
            figsize = (2,2)
        fig = plt.figure('polar',figsize=figsize)
        fig.clf()
    else:
        fig = f
    freq = params[4] / (2. * np.pi)
    theta = params[2]/np.pi*180 % 180
    stdx = np.sqrt(np.exp(params[5]))
    stdy = np.sqrt(np.exp(params[6]))
    ax = fractional_polar_axes(fig, rlim=(0, 1.2), step=(45, .4),
                               pos=pos, labelx=labelx, labely=labely)
    for ii in range(len(freq)):
        ax.plot(theta[ii], freq[ii], 'o',
                markerfacecolor=color,
                markeredgecolor='k',
                alpha=.4,
                ms=np.sqrt(stdx[ii] * stdy[ii]))
    ax.tick_params(labelsize=styles.ticklabel_fontsize)
    ax.tick_params(pad=0)

    if save_path is not None:
        plt.savefig(save_path)


def plot_GaborFit_envelope(params, model, save_path=None,
                           ax=None, f=None, figsize=None,
                           pos=None, labelx=True, labely=True):
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
    color = styles.colors[model]
    if figsize is None:
        figsize = (2,2)
    if ax is None and f is None:
        fig = plt.figure('envelope',figsize=figsize)
        fig.clf()
        ax = plt.axes([.15,.15,.8,.8])
    elif ax is None:
        ax = f.add_subplot(*pos)
    varxs  = (np.exp(params[5]))
    varys  = (np.exp(params[6]))
    freq = 5. * params[4] / (2. * np.pi)
    for ii in range(len(varxs)):
        varx  = varxs[ii]
        vary  = varys[ii]
        plt.plot(varx, vary, 'o', ms=freq[ii], mew=1,
                 markerfacecolor=color,
                 markeredgecolor='k', alpha=.4)
    ax.set_xlim(1e-0,2e1)
    ax.set_ylim(1e-0,2e1)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.yaxis.set_label_position('right')
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.minorticks_off()
    ax.tick_params(labelsize=styles.ticklabel_fontsize)
    ax.tick_params(pad=0)
    if labelx:
        ax.set_xlabel(r'var[$\parallel$]', labelpad=-1, fontsize=styles.label_fontsize)
    else:
        ax.set_xticklabels([])
    if labely:
        ax.set_ylabel(r'var[$\perp$]', labelpad=-5, fontsize=styles.label_fontsize)
    else:
        ax.set_yticklabels([])
    if save_path is not None:
        plt.savefig(save_path)


def recovery_vs_lambda(models, keep_models, results, null_results, lambdas,
                       priors, n_prior, axes, 
                       add_ylabel=False):
    labelpad = 0
    if isinstance(axes, list):
        plot_3 = True
        assert len(axes) == 3
        ae, am, ap = axes
    else:
        plot_3 = False
        ap = axes

    p = priors[n_prior]
    ii = n_prior
    for jj, m in enumerate(keep_models):
        jjp = models.index(m)
        if m in styles.models:
            fmt = styles.line_styles[m]

            color = styles.colors[m]
            label = styles.labels[m]

            if m == 'SM':
                delta = np.tile(np.nanmean(results[ii, jjp, 0, :, 0]), lambdas.size)
                mma = np.tile(np.nanmean(results[ii, jjp, 0, :, 1]), lambdas.size)
                null_delta = np.tile(np.nanmean(null_results[ii, jjp, 0, :, 0]), lambdas.size)
                null_mma = np.tile(np.nanmean(null_results[ii, jjp, 0, :, 1]), lambdas.size)
            else:
                delta = np.nanmean(results[ii, jjp, :, :, 0], axis=1)
                mma = np.nanmean(results[ii, jjp, :, :, 1], axis=1)
                null_delta = np.nanmean(null_results[ii, jjp, :, :, 0], axis=1)
                null_mma = np.nanmean(null_results[ii, jjp, :, :, 1], axis=1)
            if plot_3:
                ae.semilogx(lambdas, delta, fmt, label=label, c=color,
                        lw=styles.lw,
                            path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
                
                am.semilogx(lambdas, mma, fmt, label=label, c=color,
                        lw=styles.lw,
                            path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
            
            ap.semilogx(lambdas, .5*(delta/null_delta + mma/null_mma), fmt,
                    label=label, c=color, lw=styles.lw,
                        path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
            """
            if m != 'SM':
                if plot_3:
                    ae.semilogx(lambdas, delta, '.', c=color, ms=10, markeredgecolor='k')
                    am.semilogx(lambdas, mma, '.', c=color, ms=10, markeredgecolor='k')
                ap.semilogx(lambdas, .5*(delta/null_delta + mma/null_mma), '.', c=color,
                            ms=10, markeredgecolor='k')
                            """
    if plot_3:
        ae.set_title(p)
        ae.set_ylim([0, 50])
        am.set_title(p)
        am.set_ylim([0, 65])
        ap.set_title(p)
        if add_ylabel:
            ae.set_ylabel(r'$\Delta P$', labelpad=labelpad,
                          fontsize=styles.label_fontsize)
            am.set_ylabel(r'median($p_{\mathrm{min}}$)', labelpad=labelpad,
                          fontsize=styles.label_fontsize)
    if add_ylabel:
        ap.set_ylabel('Normalized Mean', labelpad=labelpad,
                      fontsize=styles.label_fontsize)
    ap.set_ylim([0, 1.1])
    ap.set_yticks([0, 1])
    ap.set_yticklabels([0, 1], fontsize=styles.ticklabel_fontsize)
    ap.tick_params(labelsize=styles.ticklabel_fontsize)
    ap.tick_params(pad=0)

    if not plot_3:
        axes = [ap]
    for ax in axes:
        ax.minorticks_off()
        ax.get_yaxis().set_tick_params(direction='out')
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.set_xlabel(r'$\lambda$', labelpad=labelpad,
                      fontsize=styles.label_fontsize)


def recovery_vs_oc_or_k(models, keep_models, base_folder, n_mixtures,
                        vs_OC, ax, priors, n_prior, OCs=None, keep_OCs=None, OC_k=None,
                        ks=None, k_OC=None, legend=False, keep_max=False,
                        add_ylabel=False, add_xlabel=False,
                        axlabel=None):
    labelpad = 0
    p = priors[n_prior]
    ii = n_prior
    if vs_OC:
        assert OCs is not None
        assert keep_OCs is not None
        raw = OCs
        keep = keep_OCs
        xlabel = 'Overcompleteness'
    else:
        assert ks is not None
        raw = ks
        keep = ks
        xlabel = r'$k$-sparseness'

    x = np.array([float(item) for item in keep])
    y = np.zeros((len(keep_models), x.size))
    y_std = np.zeros_like(y)
    for ii, item in enumerate(keep):
        iik = raw.index(item)
        if vs_OC:
            OC = item
            k = OC_k
        else:
            OC = k_OC
            k = item

        results, null_results, lambdas = analysis.comparison_analysis_postprocess(base_folder,
                                                                                  n_mixtures,
                                                                                  OC, k,
                                                                                  priors, keep_max)
        for kk, m in enumerate(keep_models):
            kkp = models.index(m)
            if m == 'SM':
                mean_null = np.nanmean(null_results[n_prior, kkp, 0], axis=0, keepdims=True)
                r = results[n_prior, kkp, 0] / mean_null
                pos = np.nanmean(r, axis=0).sum()
                std = np.nanstd(r.sum(axis=-1))
            else:
                mean_null = np.nanmean(null_results[n_prior, kkp], axis=1, keepdims=True)
                r = results[n_prior, kkp] / mean_null
                r_mean = np.nanmean(r, axis=1).sum(axis=1)
                r_min_idx = r_mean.argmin()
                pos = np.nanmean(r[r_min_idx], axis=0).sum()
                std = np.nanstd(r[r_min_idx].sum(axis=-1))
            y[kk, ii] = pos
            y_std[kk, ii] = std

    for ym, y_stdm, m in zip(y, y_std, keep_models):
        fmt = styles.line_styles[m]
        color = styles.colors[m]
        label = styles.labels[m]
        ax.errorbar(x, ym/2., yerr=y_stdm/np.sqrt(10), fmt=fmt, color=color,
                lw=styles.lw,
                 path_effects=[pe.Stroke(linewidth=styles.lw+1, foreground='k'), pe.Normal()])
        ax.plot(-1, -1, fmt, color=color, label=label, lw=2,
                 path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
    if vs_OC:
        ax.set_xticks(np.linspace(1, 4, 4))
        ax.set_xticklabels(np.arange(1, 4), fontsize=styles.ticklabel_fontsize)
        ax.set_xlim(.9, 3.6)
    else:
        ax.set_xlim(1.5, 16.5)
        ax.set_xticks([2, 9, 16])

    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=styles.ticklabel_fontsize)

    if add_xlabel:
        ax.set_xlabel(xlabel, labelpad=labelpad,
                      fontsize=styles.label_fontsize)
    if add_ylabel:
        ax.set_ylabel('Normalized Mean', labelpad=labelpad, fontsize=styles.label_fontsize)

    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(-1.7, 1.3), frameon=False,
                   fontsize=styles.legend_fontsize)

    ax.minorticks_off()
    ax.get_yaxis().set_tick_params(direction='out')
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(pad=0)
