from __future__ import division
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import OrderedDict
from numpy.linalg import *
from scipy.optimize import minimize
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

__authors__ = "Jesse Livezey, Alex Bujan"


def sgd(params, grads, learning_rate):
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad

def momentum(params, grads, learning_rate, momentum=.9,
             nesterov=True):
    """
    Nesterov momentum based on pylearn2 implementation.
    """
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        v = theano.shared(0.*param.get_value())
        updates[v] = v * momentum - learning_rate * grad
        delta = vel
        if nesterov:
            delta = momentum * delta - learning_rate * grad
        updates[param] = param + delta

def adam(params, grads, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    t_0 = theano.shared(np.array(0.).astype('float32'))
    one = T.constant(1)

    t = t_0 + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, grad in zip(params, grads):
        m_prev = theano.shared(0.*param.get_value())
        v_prev = theano.shared(0.*param.get_value())

        m_t = beta1*m_prev + (one-beta1)*g_t
        v_t = beta2*v_prev + (one-beta2)*g_t**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_0] = t
    
def cost(degeneracy, Wn, W_T, lambd=None, a=None, p=None):
    """
    Create costs and intermediate values from input variables.
    """
    S_T = T.dot(Wn, X_T)
    X_hat_T = T.dot(Wn.T,S_T)
    gram = T.dot(Wn,Wn.T)
    gram_diff = gram-T.eye(n_sources)
    loss = 0.

    if lambd is not None:
        if lambd == 0. or lambd > 0.:
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
            loss += error
        if lambd > 0.:
            penalty = T.log(T.cosh(S_T)).sum(axis=0).mean()
            loss += lambd * penalty
    else:
        loss = None
        error = None
        penalty = None

    X_T_normed = X_T/T.sqrt((X_T**2).sum(axis=0, keepdims=True))
    X_hat_T_normed = X_hat_T/T.sqrt((X_hat_T**2).sum(axis=0, keepdims=True))
    mse = ((X_T_normed-X_hat_T_normed)**2).sum(axis=0).mean()
    return loss, error, penalty, mse, S_T, X_hat_T

def setup_transforms(n_sources, n_mixtures):
    """
    Create transform_f and reconstruct_f functions.
    """
    X_T = theano.shared(np.zeros((1, 1), dtype='float32'))
    Wv = T.dvector('W')
    W  = T.reshape(Wv,(n_sources, n_mixtures)).astype('float32')
    epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
    W_norm = T.sqrt(epssumsq)
    Wn = W / W_norm
    
    loss, error, penalty, mse, S_T, X_hat_T = cost(degeneracy, Wn, X_T)

    transform_f = theano.function(inputs=[W], outputs=[S_T])
    reconstruct_f = theano.function(inputs=[W], outputs=[X_hat_T])
    return transform_f, reconstruct_f

def setup_lbfgsb(n_sources, n_mixtures, degeneracy, lambd):
    """
    L-BFGS-B Optimization
    """
    X_T = theano.shared(np.zeros((1, 1), dtype='float32'))
    Wv = T.dvector('W')
    W  = T.reshape(Wv,(n_sources, n_mixtures)).astype('float32')
    epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
    W_norm = T.sqrt(epssumsq)
    Wn = W / W_norm
    
    """
    Setup
    """
    loss, error, penalty, mse, S_T, X_hat_T = cost(degeneracy, Wn, X_T, lambd)
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
    return X_T, f_df, callback_f

def fit_lbfgsb(X_shared, f_df, fallback_f, data, components_):
    """
    Fit components_ from data.
    """
    def callback(w):
        res = callback_f(w)
        print('Loss: {}, Error: {}, Penalty: {}, MSE: {}'.format(*res[:4]))
    X_shared.set_value(data.astype('float32'))
    w = components_.ravel()
    w = w.reshape((self.n_sources, self.n_mixtures))
    res = minimize(self.f_df, w, jac=True, method='L-BFGS-B', callback=callback)
    w_f = res.x
    l, g = self.f_df(w_f)
    print('ICA with L-BFGS-B done!')
    print('Final loss value: {}'.format(l))
    return w_f.reshape((self.n_sources,self.n_mixtures))

def setup_sgd(n_sources, n_mixtures, lambd, degeneracy, learning_rule):
    """
    SGD optimization
    """
    X_T = T.matrix('X')
    W  = theano.shared(np.random.randn(n_sources, n_mixtures)).astype('float32'))
    epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
    W_norm = T.sqrt(epssumsq)
    Wn = W / W_norm
    
    loss, error, penalty, mse, S_T, X_hat_T = cost(degeneracy, Wn, X_T, lambd)
    loss_grad = T.grad(loss, W)
    updates = learning_rule([W], [loss_grad])

    # For monitoring
    X_T_normed = X_T/T.sqrt((X_T**2).sum(axis=0, keepdims=True))
    X_hat_T_normed = X_hat_T/T.sqrt((X_hat_T**2).sum(axis=0, keepdims=True))
    mse = ((X_T_normed-X_hat_T_normed)**2).sum(axis=0).mean()

    self.train_f = theano.function(inputs=[X_T],
                                   outputs=[loss, error, penalty, mse],
                                   updates=updates)
    return W, train_f

def fit_sgd(W, train_f, data, components_,
            tol=1e-4, batch_size=128, n_epochs=10,
            seed=20160615):
    """
    Fit components_ from data.
    """
    n_examples = data.shape[1]
    rng = np.random.RandomState(seed)
    W.set_value(components.astype('float32'))
    n_batches = n_examples//batch_size
    if n_batches * batch_size < n_examples:
        n_batches += 1

    lowest_cost = np.inf
    cur_cost = None
    decay_time = n_batches//3
    decay = np.exp(-1./decay_time)

    for ii in range(n_epochs):
        order = rng.permutation(n_examples)
        for jj in range(n_batches):
            start = jj * batch_size
            end = (jj + 1) * batch_size
            res = train_f(data[:, order[start:end]])
            print('Loss: {}, Error: {}, Penalty: {}, MSE: {}'.format(*res))
            if cur_cost is None:
                cur_cost = res[0]
            else:
                cur_cost = cur_cost*decay+res[0]*(1-decay)
        if cur_cost < lowest_cost-tol:
            lowest_cost = cur_cost
        else:
            break

    print('ICA with SGD done!')
    print('Final loss value: {}'.format(cur_cost))
    return W.get_value()

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
                 a=None, optimizer='L-BFGS-B', learning_rule=None,
                 **fit_kwargs):

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
        if optimizer = 'L-BFGS-B':
            self.fit_info = setup_lbfgsb(n_sources, n_mixtures, degeneracy, lambd)
            self.fit_f = fit_lbfgsb
        elif optimizer = 'sgd':
            self.fit_info = setup_sgd(n_sources, n_mixtures, lambd,
                                      degeneracy, learning_rule)
            self.fit_f = fit_sgd
        else:
            raise ValueError

        self.transform_f, self.reconstruct_f = setup_transforms(n_sources,
                                                                n_mixtures)

    def fit(self, X, y=None):
        """
        Fit an ICA model to data.

        Parameters
        ----------
        X : ndarray
            Data array (mixtures by samples)
        """
        self.components_ = w/norm(w, axis=-1, keepdims=True)
        self.fit_f(*self.fit_info, X, self.components_,
                   **self.fit_kwargs)
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
