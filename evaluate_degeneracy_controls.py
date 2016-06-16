import numpy as np
from numpy.linalg import norm
import ica_classes as ica_c
reload(ica_c)

overcompleteness = 2
n_mixtures = 64
n_sources = n_mixtures*overcompleteness
initial_conditions = ['random','collimated','pathological']
degeneracy_controls = ['L2','COULOMB','RANDOM','L4','COULOMB_A']

def decorr(X):
    return (3./2.)*X - (1./2.)*X.dot(X.T).dot(X)

def get_Winit(n_sources, n_mixtures, init=random):
    if init=='random':
        w = np.random.randn(n_sources, n_mixtures)
    elif init=='collimated':
        w = np.tile(np.random.randn(1, n_mixtures), (n_sources, 1))+\
            np.random.randn(n_sources, n_mixtures)/2.
    elif init=='pathological':
        w = np.tile(np.eye(n_mixtures), (n_sources//n_mixtures, 1))+\
            np.random.randn(n_sources, n_mixtures)/100.
    w = w/norm(w_0, axis=-1, keepdims=True)
    return w

def get_W(initial_conditions, degeneracy_controls, n_sources, n_mixtures):
    """Obtain W that minimizes the ICA loss funtion without the penalty"""
    X = np.ones((n_mixtures, 2), dtype='float32')
    W_0 = np.zeros((len(initial_conditions),n_sources,n_mixtures))
    W = np.zeros((len(initial_conditions),len(degeneracy_controls),n_sources,n_mixtures))
    for i,init in enumerate(initial_conditions):
        W_0[i] = get_Winit(n_sources, n_mixtures, init=init)
        for j,dg in enumerate(degeneracy_controls):
            if dg!='QUASI-ORTHO':
                ica = ica_c.ICA(n_mixtures=n_mixtures, n_sources=n_sources, 
                                degeneracy=dg, lambd=0., w_init=W_0[i].copy())
                W[i,j] = ica.fit(X).components_
            else:
                wd = decorr(W_0[i].copy())
                wd = wd/norm(wd, axis=-1, keepdims=True)
                W[i,j] = wd
    return W, W_0

def compute_angles(w):
    w = w/norm(w, axis=-1, keepdims=True)
    gram = w.dot(w.T)
    gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
    return np.arccos(abs(gram_off_diag))/np.pi*180.
