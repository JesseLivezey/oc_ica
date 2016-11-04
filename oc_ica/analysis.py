from __future__ import division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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
            rng.randn(n_sources, n_mixtures)/100
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
            if dg!='QUASI-ORTHO':
                W[i,j] = get_W(W_0[i], dg, rng=rng, **kwargs)
            else:
                W[i,j] = quasi_ortho_decorr(W_0[i])
    return W, W_0

def compute_angles(w):
    w = normalize_W(w)
    gram = w.dot(w.T)
    gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
    return cos2deg(gram_off_diag)

def cos2deg(cos):
    return np.arccos(abs(cos))/np.pi*180.
