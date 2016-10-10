from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import OrderedDict
from scipy.optimize import minimize
import theano
import theano.tensor as T

from oc_ica.optimizers import ica_optimizers

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
    prior : str
        'soft' (soft L1) or 'hard' (L1)
    rng : numpy.RandomState
        Randomstate for initialization.
    a : int
        Inverse power for Coulomb cost.
    optimizer : str
        Type of optimizer. 'L_BFGS-B' or 'sgd'.
    learning_rule : str
        Type of learning rule for sgd.
    fit_kwargs : str
        kwargs for optimizer.
    """
    def __init__(self, n_mixtures, n_sources=None, lambd=1e-2,
                 w_init=None, degeneracy=None, p=None, prior='soft', rng=None,
                 optimizer='L-BFGS-B', learning_rule=None,
                 **fit_kwargs):

        if learning_rule is not None:
            assert optimizer == 'sgd'

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

        self.fit_kwargs = fit_kwargs

        if w_init is None:
            w_0 = rng.randn(n_sources, n_mixtures)
        else:
            w_0 = w_init
        w_0 = w_0/np.sqrt(np.maximum((w_0**2).sum(axis=1, keepdims=True), 1e-7))
        self.n_mixtures = n_mixtures
        self.n_sources = n_sources
        self.components_ = w_0
        if optimizer == 'L-BFGS-B':
            self.optimizer = ica_optimizers.LBFGSB(n_sources=n_sources, n_mixtures=n_mixtures,
                                               degeneracy=degeneracy,
                                               lambd=float(lambd), p=p, prior=prior)
        elif optimizer == 'sgd':
            self.optimizer = ica_optimizers.SGD(n_sources=n_sources, n_mixtures=n_mixtures,
                                            w_0=w_0, lambd=float(lambd),
                                            degeneracy=degeneracy,
                                            learning_rule=learning_rule,
                                            p=p, prior=prior)
        else:
            raise ValueError

    def _normalize_components(self):
        """
        Normalize components_ to unit norm.
        """
        self.components_ = self.components_/np.linalg.norm(self.components_, axis=-1, keepdims=True)

    def fit(self, X, y=None):
        """
        Fit an ICA model to data.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)
        """
        self._normalize_components()
        w_f = self.optimizer.fit(X, self.components_)
        self.components_ = w_f
        self._normalize_components()
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
        self._normalize_components()
        sources = self.optimizer.transform(X, self.components_)
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
        self._normalize_components()
        self.fit(X)
        self._normalize_components()
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
        self._normalize_components()
        X_hat = self.optimizer.reconstruct(X, self.components_)
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
        self._normalize_components()
        losses = self.optimizer.losses(X, self.components_)
        return losses
