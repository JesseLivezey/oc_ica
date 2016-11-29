from __future__ import division
import numpy as np
import scipy as sp
from collections import OrderedDict
from scipy.optimize import minimize
import theano
import theano.tensor as T

from oc_ica.optimizers.ica_optimizers import Optimizer

__authors__ = "Jesse Livezey, Alex Bujan"


class SC_Optimizer(Optimizer):
    """
    Optimizer.
    """
    def __init__(self, lambd, **fit_kwargs):
        self.lambd = lambd
        super(SC_Optimizer, self).__init__(**fit_kwargs)

    def fit(self, data, components_):
        """
        Fit components_ from data.
        """
        def callback(w):
            res = self.callback_f(w.astype('float32'))
            print('Loss: {}, Error: {}, Penalty: {}, MSE: {}'.format(*res[:4]))
        if self.verbose:
            cb = callback
        else:
            cb = None
        self.X.set_value(data.astype('float32'))
        self.components_shape = components_.shape
        float_f_df = lambda w: self.f_df(w)
        w = components_.ravel()
        res = minimize(float_f_df, w, jac=True, method='L-BFGS-B',
                       callback=cb)
        w_f = res.x
        l, g = float_f_df(w_f)
        print('SC with L-BFGS-B done!')
        print('Final loss value: {}'.format(l))
        return w_f.reshape(components_.shape)

    @classmethod
    def reconstruction_cost(cls, W, X, S):
        return .5*((X-W.T.dot(S))**2).mean(axis=1).sum()

    def prior_cost(self, S):
        raise NotImplementedError


class SC_Hard(SC_Optimizer):
    """
    Optimizer.
    """
    def __init__(self, lambd, **fit_kwargs):
        self.L = theano.shared(np.array(1.).astype('float32'))
        super(SC_Optimizer, self).__init__(lambd, **fit_kwargs)

    def f_df(self, w):
        self.reset_L(w.reshape(self.components_shape))
        val = self.f_df_f(w.astype('float32'))
        return [r.astype('float64') for r in val]

    def reset_L(self, components_):
        W = components_/np.linalg.norm(components_, axis=-1, keepdims=True)
        gram = W.dot(W.T)
        n = gram.shape[0]-1
        L = 2. * sp.linalg.eigh(gram, eigvals_only=True, eigvals=(n, n))[0]
        self.L.set_value(np.array(L).astype('float32'))

    def setup(self, n_sources, n_mixtures):
        """
        L-BFGS-B Optimization
        """
        X = theano.shared(np.zeros((1, 1), dtype='float32'))
        self.X = X
        Wv = T.vector('W')
        W  = T.reshape(Wv,(n_sources, n_mixtures))
        epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
        W_norm = T.sqrt(epssumsq)
        Wn = W / W_norm

        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X)
        loss_grad = T.grad(error, Wv)

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()

        self.f_df_f = theano.function(inputs=[Wv], outputs=[loss, loss_grad])
        if error is None:
            error = 0.*loss
        if penalty is None:
            penalty = 0.*loss
        self.callback_f = theano.function(inputs=[Wv],
                                          outputs=[loss, error, penalty, mse])

    def prior_cost(self, S):
        return abs(S).mean(axis=1).sum()

    def transforms(self, W, X, n_steps=25):
        n_batch = X.shape[1]
        n_neurons = W.shape[0]

        S_init = T.zeros((n_neurons, n_batch))
        S_init2 = T.zeros((n_neurons, n_batch))
        t = T.ones(1)

        def step(x_km1, y_k, t, W, X, lambd, L):
            rec_grad = T.grad(self.reconstruction_cost(W, X, y_k), y_k)
            y_k_prime = y_k - (X.shape[1].astype('float32')*rec_grad/L)
            abs_y_k_prime = abs(y_k_prime) - lambd/L
            x_k = T.sgn(y_k_prime) * T.nnet.relu(abs_y_k_prime)

            t1 = 0.5 * (1 + T.sqrt(1. + 4. * t ** 2))
            y_kp1 = x_k + (t - 1.) * (x_k - x_km1) / t1
            return x_k, y_kp1, t1

        outputs, updates = theano.scan(step,
                                       outputs_info=[S_init, S_init2, t],
                                       non_sequences=[W, X, self.lambd, self.L],
                                       n_steps=n_steps)
        xt, yt, tt = outputs
        S = xt[-1]

        X_hat = W.T.dot(S)
        return S, X_hat

    def setup_transforms(self, **kwargs):
        """
        Create transform_f and reconstruct_f functions.
        """
        X = T.matrix('X')
        W = T.matrix('W')
        epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
        W_norm = T.sqrt(epssumsq)
        Wn = W / W_norm
        
        S, X_hat = self.transforms(Wn, X)

        self.transform_f = theano.function(inputs=[X, W], outputs=[S])
        self.reconstruct_f = theano.function(inputs=[X, W], outputs=[X_hat])

        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, **kwargs)
        outputs = [loss, error, penalty, mse]
        outputs = [o if o is not None else 0.* loss for o in outputs]

        self.losses_f = theano.function(inputs=[X, W],
                                        outputs=outputs)

    def transform(self, X, W):
        self.reset_L(W)
        return self.transform_f(X.astype('float32'), W.astype('float32'))

    def reconstruct(self, X, W):
        reset_L(W)
        return self.reconstruct_f(X.astype('float32'), W.astype('float32'))

    def losses(self, X, W):
        reset_L(W)
        return self.losses_f(X.astype('float32'), W.astype('float32'))

    def cost(self, Wn, X, **kwargs):
        """
        Create costs and intermediate values from input variables.
        """
        S, X_hat = self.transforms(Wn, X)
        S = theano.gradient.disconnected_grad(S)
        error = self.reconstruction_cost(Wn, X, S)
        penalty = self.prior_cost(S)
        loss = error + self.lambd * penalty

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()
        return loss, error, penalty, mse, S, X_hat


class SC_Soft(SC_Optimizer):
    """
    Optimizer.
    """
    def f_df(self, w):
        s0 = np.zeros(self.n_sources, self.X.values.shape[0])
        self.Wv.set_value(w.astype('float32'))
        res = minimize(self.f_dfds_f, s0, jac=True, method='L-BFGS-B',
                       callback=cb)
        s = res.x
        val = self.f_dfdw_f(w.astype('float32'), s.astype('float32'))
        return [r.astype('float64') for r in val]

    def setup(self, n_sources, n_mixtures):
        """
        L-BFGS-B Optimization
        """
        self.n_sources = n_sources
        self.n_mixtures = n_mixtures
        X = theano.shared(np.zeros((1, 1), dtype='float32'))
        self.X = X
        Sv = T.vector('S')
        S = T.reshape(Sv, (X.shape[0], n_sources))
        Wv = T.vector('W')
        W  = T.reshape(Wv,(n_sources, n_mixtures))
        epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
        W_norm = T.sqrt(epssumsq)
        Wn = W / W_norm

        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, S)
        loss_grad = T.grad(error, Wv)

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()

        self.f_dfdw_f = theano.function(inputs=[Wv, Sv], outputs=[loss, loss_grad])
        if error is None:
            error = 0.*loss
        if penalty is None:
            penalty = 0.*loss
        self.callback_f = theano.function(inputs=[Wv],
                                          outputs=[loss, error, penalty, mse])

        Wv = theano.shared(np.zeros(1, dtype='float32'))
        self.Wv = W
        epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
        W_norm = T.sqrt(epssumsq)
        Wn = W / W_norm
        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, S)
        loss_grad_S = T.grad(error, Sv)
        self.f_dfds_f = theano.function(inputs=[Sv], outputs=[loss, loss_grad_S])

    def prior_cost(self, S):
        return T.log(T.cosh(S)).mean(axis=1).sum()

    def cost(self, Wn, X, S, **kwargs):
        """
        Create costs and intermediate values from input variables.
        """
        X_hat = S.dot(X)
        error = self.reconstruction_cost(Wn, X, S)
        penalty = self.prior_cost(S)
        loss = error + self.lambd * penalty

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()
        return loss, error, penalty, mse, S, X_hat
