from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
from numpy.linalg import *
from scipy.optimize import minimize

__authors__ = "Jesse Livezey, Alex Bujan"

    
class ICA(BaseEstimator, TransformerMixin):
    """
    ICA class.

    Parameters
    ----------
    n_mixtures : int
        Dimensionality of mixtures.
    n_sources : int
        Dimensionality of sources.
    lambd : float
        Coefficient for sparse penalty.
    w_init : ndarray
        Initialization for filter matrix.
    degeneracy : str
        Type of degeneracy control to use.
    p : int
        p for 'Lp' norm degeneracy control.
    rng : numpy.RandomState
        Randomstate for initialization.
    """
    def __init__(self, n_mixtures, n_sources=None, lambd=1e-2,
                 w_init=None, degeneracy='RICA', p=None, rng=None,
                 a=None):

        if rng is None:
            seed = np.random.randint(100000)
            print('Random seed: {}'.format(seed))
            rng = np.random.RandomState(seed)

        if n_sources==None:
            n_sources = n_mixtures
            print('Complete ICA')
        else:
            if n_sources > n_mixtures:
                print('Overcomplete ICA')
        print('Degeneracy control: {}'.format(degeneracy))

        if w_init is None:
            w_0 = rng.randn(n_sources, n_mixtures)
        else:
            w_0 = w_init
        w_0 = w_0/np.sqrt(np.maximum((w_0**2).sum(axis=1, keepdims=True), 1e-7))
        self.n_mixtures = n_mixtures
        self.n_sources = n_sources
        self.components_ = w_0

        """
        # L-BFGS Optimization
        """
        X_T = theano.shared(np.zeros((1, 1), dtype='float32'))
        self.X = X_T
        Wv = T.dvector('W')
        W  = T.reshape(Wv,(n_sources,n_mixtures)).astype('float32')
        epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
        W_norm = T.sqrt(epssumsq)
        Wn = W / W_norm
        
        """
        Setup
        """
        S_T = T.dot(Wn, X_T)
        X_hat_T = T.dot(Wn.T,S_T)
        gram = T.dot(Wn,Wn.T)
        gram_diff = gram-T.eye(n_sources)
        if degeneracy == 'RICA':
            error = .5 * T.sum((X_hat_T-X_T)**2, axis=0).mean()
        elif degeneracy == 'L2':
            error = (1./2) * T.sum(gram_diff**2)
        elif degeneracy == 'L4':
            error = (1./4) * T.sum((gram_diff**2)**2)
        elif degeneracy == 'Lp':
            assert isinstance(p, int)
            assert (p % 2) == 0
            error = gram_diff
            for ii in range(p//2):
                error = error**2
            #error = (1./2**p) * T.sum(error)
            error = (1./p) * T.sum(error)
        elif degeneracy == 'COULOMB':
            epsilon = 0.1
            error = .5 * T.sum(1. / T.sqrt(1. + epsilon - gram**2))
        elif degeneracy == 'COULOMB_A':
            assert a is not None
            epsilon = 0.1
            error = .5 * T.sum((1. / (1. + epsilon - gram**2)**(1 / a)) - (gram**2 / a))
        elif degeneracy == 'RANDOM':
            epsilon = 0.1
            error = -.5 * T.sum(T.log(1. + epsilon - gram**2))
        elif degeneracy == 'RANDOM_A':
            epsilon = 0.1
            error = -.5 * T.sum(T.log(1. + epsilon - gram**2) - gram**2)
        else:
            raise ValueError
        penalty = T.log(T.cosh(S_T)).sum(axis=0).mean()
        loss =  error + lambd * penalty
        loss_grad = T.grad(loss, Wv)
        # For monitoring
        X_T_normed = X_T/T.sqrt((X_T**2).sum(axis=0, keepdims=True))
        X_hat_T_normed = X_hat_T/T.sqrt((X_hat_T**2).sum(axis=0, keepdims=True))
        mse = ((X_T_normed-X_hat_T_normed)**2).sum(axis=0).mean()

        """
        Training
        """
        self.f_df = theano.function(inputs=[Wv], outputs=[loss.astype('float64'),loss_grad.astype('float64')])
        self.callback_f = theano.function(inputs=[Wv],
                                     outputs=[loss, error, penalty, mse])
        self.transform_f = theano.function(inputs=[W], outputs=[S_T])
        self.reconstruct_f = theano.function(inputs=[W], outputs=[X_hat_T])

    def fit(self, X, y=None):
        """
        Fit an ICA model to data.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)
        """
        def callback(w):
            res = self.callback_f(w)
            print('Loss: {}, Error: {}, Penalty: {}, MSE: {}'.format(*res[:4]))
        self.X.set_value(X.astype('float32'))
        w = self.components_.ravel()
        w = w.reshape((self.n_sources, self.n_mixtures))
        res = minimize(self.f_df, w, jac=True, method='L-BFGS-B', callback=callback)
        w_f = res.x
        l, g = self.f_df(w_f)
        print('ICA with L-BFGS-B done!')
        print('Final loss value: {}'.format(l))
        w = w_f.reshape((self.n_sources,self.n_mixtures))
        self.components_ = w/norm(w, axis=-1, keepdims=True)
        return self

    def transform(self, X, y=None):
        """
        Transform data into sources.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)

        Returns
        -------
        sources : ndarray
            Data transformed into sources.
        """
        self.X.set_value(X.astype('float32'))
        sources = self.transform_f(self.components_.astype('float32'))[0]
        return sources

    def fit_transform(self, X, y=None):
        """
        Fit an ICA model to data and returns transformed data.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)

        Returns
        -------
        sources : ndarray
            Data transformed into sources.
        """
        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X):
        """
        Reconstruct data with model.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)

        Returns
        -------
        X_hat : ndarray
            Data transformed into sources.
        """
        self.X.set_value(X.astype('float32'))
        X_hat = self.reconstruct_f(self.components_.astype('float32'))[0]
        return X_hat

    def losses(self, X):
        """
        Return the loss, error, penalty, and mse.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)

        Returns
        -------
        losses : ndarray
            Loss for each data sample.
        """
        self.X.set_value(X.astype('float32'))
        losses = self.callback_f(self.components_.ravel().astype('float32'))
        return losses
