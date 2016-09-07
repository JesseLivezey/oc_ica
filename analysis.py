from __future__ import division
import numpy as np
import scipy as sp
from models import ica
reload(ica)


def find_max_allowed_k(As, n_sources):
    """
    Given either a dictionary containing lists of mixing
    matrices or a list of matrices, compute the highest
    k-sparseness that will still allow recovery.
    """
    def max_allowed_for_list(A_list):
        k = n_sources
        for A in A_list:
            A = A/np.linalg.norm(A, axis=0, keepdims=True)
            off_gram = A.T.dot(A) - np.eye(A.shape[1])
            angles = compute_angles(A.T)
            mu = abs(off_gram).max()
            k_temp = int(np.floor(.5*(1. + 1./mu)))
            if k_temp < k:
                k = k_temp
            A = np.linalg.pinv(A).T
            A = A/np.linalg.norm(A, axis=0, keepdims=True)
            off_gram = A.T.dot(A) - np.eye(A.shape[1])
            angles = compute_angles(A.T)
            mu = abs(off_gram).max()
            k_temp = int(np.floor(.5*(1. + 1./mu)))
        return k

    if isinstance(As, dict):
        k = n_sources
        for key in sorted(As.keys()):
            k_temp = max_allowed_for_list(As[key])
            if k_temp < k:
                k = k_temp
    elif isinstance(As, list):
        k = max_allowed_for_list(As)
    else:
        raise ValueError
    
    return k

def recovery_statistics(W, W0):
    """
    Compute recovery statistics for mixing matrix and
    recovered matrix.
    """
    def hellinger(p, q):
        return np.linalg.norm(np.sqrt(p) - np.sqrt(q))

    def perm_delta(W, W0):
        P = abs(W.dot(W0.T))
        P_max = np.zeros_like(P)
        max_idx = np.array([np.arange(P.shape[0]), np.argmax(P, axis=1)])
        P_max[max_idx] = 1.
        return abs(P_max.sum(axis=0)-1).sum()

    w_angles = compute_angles(W)
    w0_angles = compute_angles(W0)
    bins = np.arange(0, 91)
    w_dist = np.histogram(w_angles, bins, density=True)[0]
    w0_dist = np.histogram(w0_angles, bins, density=True)[0]

    dist_val = hellinger(w_dist, w0_dist)
    if dist_val==np.inf:
       dist_val = dist(w0_dist, w_dist)

    return dist_val, perm_delta(W, W0)

def decorr_complete(X):
    return np.linalg.inv(np.sqrt(X.dot(X.T))).dot(X)
#    return X.dot(X.T.dot(X)**(-1/2))

def decorr(w):
    return (3/2) * w - (1/2) * w.dot(w.T).dot(w)

def get_Winit(n_sources, n_mixtures, init='random'):
    if init=='random':
        w = np.random.randn(n_sources, n_mixtures)
    elif init=='collimated':
        w = np.tile(np.random.randn(1, n_mixtures), (n_sources, 1))+\
            np.random.randn(n_sources, n_mixtures)/2
    elif init=='pathological':
        w = np.tile(np.eye(n_mixtures), (n_sources//n_mixtures, 1))+\
            np.random.randn(n_sources, n_mixtures)/100
    w = w/np.linalg.norm(w, axis=-1, keepdims=True)
    return w

def quasi_ortho_decorr(w):
    wd = decorr(w.copy())
    wd = wd/np.linalg.norm(wd, axis=-1, keepdims=True)
    return wd

def get_W(w_init, degeneracy):
    """Obtain W that minimizes the ICA loss funtion without the penalty"""
    n_sources, n_mixtures = w_init.shape
    X = np.ones((n_mixtures, 2), dtype='float32')
    model = ica.ICA(n_mixtures=n_mixtures, n_sources=n_sources, 
                    degeneracy=degeneracy, lambd=0., w_init=w_init.copy())
    return model.fit(X).components_

def evaluate_dgcs(initial_conditions, degeneracy_controls, n_sources, n_mixtures):
    W_0 = np.zeros((len(initial_conditions),n_sources,n_mixtures))
    W = np.zeros((len(initial_conditions),len(degeneracy_controls),
                  n_sources,n_mixtures))
    for i,init in enumerate(initial_conditions):
        W_0[i] = get_Winit(n_sources, n_mixtures, init=init)
        for j,dg in enumerate(degeneracy_controls):
            if dg!='QUASI-ORTHO':
                W[i,j] = get_W(W_0[i], dg)
            else:
                W[i,j] = quasi_ortho_decorr(W_0[i])
    return W, W_0

def compute_angles(w):
    w = w/np.linalg.norm(w, axis=-1, keepdims=True)
    gram = w.dot(w.T)
    gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
    return np.arccos(abs(gram_off_diag))/np.pi*180
