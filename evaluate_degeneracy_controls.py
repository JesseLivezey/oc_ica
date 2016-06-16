from __future__ import division
import numpy as np
from numpy.linalg import inv,norm
import ica as ocICA
reload(ocICA)

def decorr_complete(X):
    return inv(np.sqrt(X.dot(X.T))).dot(X)
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
    w = w/norm(w_0, axis=-1, keepdims=True)
    return w

def quasi_ortho_decorr(w):
    wd = decorr(w.copy())
    wd = wd/norm(wd, axis=-1, keepdims=True)
    return wd

def get_W(w_init, degeneracy):
    """Obtain W that minimizes the ICA loss funtion without the penalty"""
    n_sources, n_mixtures = w_init.shape
    X = np.ones((n_mixtures, 2), dtype='float32')
    ica = ocICA.ICA(n_mixtures=n_mixtures, n_sources=n_sources, 
                    degeneracy=degeneracy, lambd=0., w_init=w_init.copy())
    return = ica.fit(X).components_

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
    w = w/norm(w, axis=-1, keepdims=True)
    gram = w.dot(w.T)
    gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
    return np.arccos(abs(gram_off_diag))/np.pi*180
