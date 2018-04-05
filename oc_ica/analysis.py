from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import glob, os, h5py
try:
    from importlib import reload
except ImportError:
    pass

from oc_ica.models import ica
reload(ica)


def normalize_A(A):
    return A/np.linalg.norm(A, axis=0, keepdims=True)

def normalize_W(W):
    return W/np.linalg.norm(W, axis=1, keepdims=True)


def find_max_allowed_k(As):
    """
    Given an array of mixing matrices,
    compute the highest k-sparseness
    that will still allow recovery.
    """
    shape = As.shape
    n_sources = shape[-1]
    As = As.reshape(-1, shape[-2], shape[-1])

    k = n_sources
    for A in As:
        A = normalize_A(A)
        off_gram = A.T.dot(A) - np.eye(A.shape[1])
        mu = abs(off_gram).max()
        k_temp = int(np.floor(.5*(1. + 1./mu)))
        if k_temp < k:
            k = k_temp

    return k

def hellinger(p, q):
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q))

def perm_delta(A, W, full=False):
    P = abs(W.dot(A))

    P_max = np.zeros_like(P, dtype=bool)
    P_max[np.arange(P.shape[0]), np.argmax(P, axis=1)] = 1

    max_vals = P[P_max]
    max_angles = cos2deg(max_vals)

    if full:
        other_vals = P[np.logical_not(P_max)]
        other_angles = cos2deg(other_vals)

        return (abs(P_max.sum(axis=0)-1).sum(),
                max_angles,
                other_angles)
    return (abs(P_max.sum(axis=0)-1).sum(),
            max_angles)

def angle_histogram(angles):
    bins = np.arange(0, 91)
    dist = np.histogram(angles, bins, density=True)[0]
    return dist, bins

def recovery_statistics_AW(A, W, full=False):
    """
    Compute recovery statistics for mixing matrix and
    recovered matrix.
    """
    A = normalize_A(A)
    W = normalize_W(W)
    results = perm_delta(A, W, full=full)
    if full:
        delta_p, ma, oa = results
        return delta_P, np.nanmedian(ma), ma, oa
    delta_P, ma = results
    return delta_P, np.nanmedian(ma)

def recovery_statistics2_AW(A, W):
    """
    Compute recovery statistics for mixing matrix and
    recovered matrix.
    """
    A = normalize_A(A)
    W = normalize_W(W)
    P = abs(W.dot(A))
    angles = np.full(P.shape[0], np.nan)
    for ii in range(P.shape[0]):
        x, y = np.unravel_index(P.argmax(), P.shape)
        angles[ii] = P[x, y]
        P[x] = -np.inf
        P[:, y] = -np.inf
    return np.median(angles)

def decorr_complete(X):
    return np.linalg.inv(np.sqrt(X.dot(X.T))).dot(X)
#    return X.dot(X.T.dot(X)**(-1/2))

def decorr(w):
    return (3/2) * w - (1/2) * w.dot(w.T).dot(w)

def get_Winit(n_sources, n_mixtures, init='random', rng=None):
    if rng is None:
        rng = np.random.RandomState(20160916)

    if init=='random':
        w = rng.randn(n_sources, n_mixtures)
    elif init=='collimated':
        w = np.tile(rng.randn(1, n_mixtures), (n_sources, 1))+\
            rng.randn(n_sources, n_mixtures)/2
    elif init=='pathological':
        w = np.tile(np.eye(n_mixtures), (n_sources//n_mixtures, 1))+\
            rng.randn(n_sources, n_mixtures)/100.
    w = w/np.linalg.norm(w, axis=-1, keepdims=True)
    return w

def quasi_ortho_decorr(w):
    wd = decorr(w.copy())
    wd = wd/np.linalg.norm(wd, axis=-1, keepdims=True)
    return wd

def get_W(w_init, degeneracy, rng=None, **kwargs):
    """
    Obtain W that minimizes the ICA loss funtion without the penalty
    """
    if rng is None:
        rng = np.random.RandomState(20160915)
    n_sources, n_mixtures = w_init.shape
    X = np.ones((n_mixtures, 2), dtype='float32')
    model = ica.ICA(n_mixtures=n_mixtures, n_sources=n_sources,
                    degeneracy=degeneracy, lambd=0., w_init=w_init.copy(),
                    rng=rng, **kwargs)
    return model.fit(X).components_

def evaluate_dgcs(initial_conditions, degeneracy_controls, n_sources,
                  n_mixtures, rng=None, **kwargs):
    W_0 = np.zeros((len(initial_conditions),n_sources,n_mixtures))
    W = np.zeros((len(initial_conditions),len(degeneracy_controls),
                  n_sources,n_mixtures))
    for i,init in enumerate(initial_conditions):
        W_0[i] = get_Winit(n_sources, n_mixtures, init=init, rng=rng)
        for j,dg in enumerate(degeneracy_controls):
            if dg == 'QUASI-ORTHO':
                W[i,j] = quasi_ortho_decorr(W_0[i])
            elif dg == 'INIT':
                W[i, j] = W_0[i]
            else:
                try:
                    dg = 'L' + str(int(dg))
                except ValueError:
                    pass
                W[i,j] = get_W(W_0[i], dg, rng=rng, **kwargs)
    return W, W_0

def compute_angles(w):
    angles = np.array([], dtype=float)
    if w.ndim == 2:
        w = w[np.newaxis, ...]
    for wp in w:
        wp = normalize_W(wp)
        gram = wp.dot(wp.T)
        gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
        angles = np.concatenate([angles, cos2deg(gram_off_diag)], axis=0)
    return angles

def cos2deg(cos):
    return np.arccos(abs(cos))/np.pi*180.

def comparison_analysis_postprocess(base_folder, n_mixtures, OC, k, priors,
                                    keep_max=None,
                                    overwrite=False):
    n_sources = int(n_mixtures * float(OC))
    if keep_max is None:
        fit_folder = 'comparison_mixtures-{}_sources-{}_k-{}_priors-{}'.format(n_mixtures,
                                                                               n_sources, k,
                                                                               '_'.join(priors))
    else:
        fit_folder = 'comparison_mixtures-{}_sources-{}_k-{}_priors-{}_keep_max-{}'.format(n_mixtures,
                                                                               n_sources, k,
                                                                               '_'.join(priors),
                                                                               keep_max)
    a_file = 'a_array-{}_OC-{}_priors-{}.h5'.format(n_mixtures, OC, '_'.join(priors))
    #print a_file
    #print fit_folder
    fit_files = sorted(glob.glob(os.path.join(base_folder, fit_folder,
        'comparison*.h5')))
    sc_fits = None
    models = [f.split('.')[-2].split('-')[-2].split('_keep')[0] for f in fit_files]
    #print len(models), models
    #print ''

    with h5py.File(os.path.join(base_folder, a_file), 'r') as f:
        A_array = f['A_array'].value
        A_priors = f['A_priors'].value

    with h5py.File(os.path.join(base_folder, fit_folder, fit_files[0]),
        'r') as f:
        lambdas = f['lambdas'].value
        n_mixtures, n_sources = A_array.shape[2:]
        n_iter = A_array.shape[1]

    W_fits = np.full((len(A_priors), len(models), lambdas.size, n_iter,
        n_sources, n_mixtures),
                         np.nan, dtype='float32')
    results = np.full((len(A_priors), len(models), lambdas.size, n_iter),
                          np.nan, dtype='float32')
    null_results = np.full((len(A_priors), len(models), lambdas.size,
        (n_iter**2-n_iter)//2), np.nan, dtype='float32')

    for ii, f_name in enumerate(fit_files):
        with h5py.File(os.path.join(base_folder, fit_folder, f_name),
        'r') as f:
            W_fits[:, ii] = np.squeeze(f['W_fits'])[:, :, :n_iter]

        if sc_fits is not None:
            loc = 0
            for ii, f_name in enumerate(sc_fits):
                with h5py.File(os.path.join(base_folder, fit_folder, f_name), 'r') as f:
                    n_lambdas = f['W_fits'].shape[2]
                    W_fits[:, -1, loc:loc+n_lambdas, :10] = np.squeeze(f['W_fits'])[:, :, :n_iter]
                    loc += n_lambdas

    results_file = os.path.join(base_folder, fit_folder, 'results.h5')
    if (not overwrite) and os.path.exists(results_file):
        with h5py.File(results_file) as f:
            results = f['results'].value
    else:
        for ii, p in enumerate(A_priors):
            for jj, m in enumerate(models):
                for kk, l in enumerate(lambdas):
                    for ll in range(n_iter):
                        try:
                            A = A_array[ii, ll]
                            W = W_fits[ii, jj, kk, ll]
                            assert (not np.isnan(A.sum())) and (not np.isnan(W.sum()))
                            results[ii, jj, kk, ll] = recovery_statistics2_AW(A, W)
                        except (ValueError, AssertionError):
                            pass
        with h5py.File(results_file, 'w') as f:
            f.create_dataset('results', data=results)

    results_file = os.path.join(base_folder, fit_folder, 'null_results.h5')
    if (not overwrite) and os.path.exists(results_file):
        with h5py.File(results_file) as f:
            null_results = f['null_results'].value
    else:
        for ii, p in enumerate(A_priors):
            for jj, m in enumerate(models):
                for kk, l in enumerate(lambdas):
                    loc = 0
                    for ll in range(n_iter):
                        for mm in range(ll+1, n_iter):
                            try:
                                A = A_array[ii, ll]
                                W = W_fits[ii, jj, kk, mm]
                                assert (not np.isnan(A.sum())) and (not np.isnan(W.sum()))
                                null_results[ii, jj, kk, loc] = recovery_statistics2_AW(A, W)
                                loc += 1
                            except (ValueError, AssertionError):
                                pass
        with h5py.File(results_file, 'w') as f:
            f.create_dataset('null_results', data=null_results)
    return results, null_results, lambdas
