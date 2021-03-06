from __future__ import division
import numpy as np
from scipy.optimize import minimize
import theano
import theano.tensor as T
from collections import OrderedDict

__authors__ = "Jesse Livezey, Alex Bujan"


def sgd(params, grads, learning_rate=.01):
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates

def momentum(params, grads, learning_rate=.01, momentum=.1,
             nesterov=True):
    """
    Nesterov momentum based on pylearn2 implementation.
    """
    updates = OrderedDict()
    for param, grad in zip(params, grads):
        v = theano.shared(0.*param.get_value())
        updates[v] = v * momentum - learning_rate * grad
        delta = v
        if nesterov:
            delta = momentum * delta - learning_rate * grad
        updates[param] = param + delta
    return updates

def adam(params, grads, learning_rate=0.001, beta1=0.9,
         beta2=0.999, epsilon=1e-8):

    updates = OrderedDict()
    t_0 = theano.shared(np.array(0.).astype('float32'))
    one = T.constant(1)

    t = t_0 + 1
    a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

    for param, grad in zip(params, grads):
        m_prev = theano.shared(0.*param.get_value())
        v_prev = theano.shared(0.*param.get_value())

        m_t = beta1*m_prev + (one-beta1)*grad
        v_t = beta2*v_prev + (one-beta2)*grad**2
        step = a_t*m_t/(T.sqrt(v_t) + epsilon)

        updates[m_prev] = m_t
        updates[v_prev] = v_t
        updates[param] = param - step

    updates[t_0] = t
    return updates


class Optimizer(object):
    """
    Optimizer.
    """
    def __init__(self, prior='soft', verbose=False, **fit_kwargs):
        #assert prior in ['soft', 'hard']
        self.verbose = verbose
        self.prior = prior
        self.fit_kwargs = fit_kwargs
        self.setup(**fit_kwargs)
        self.setup_transforms(**fit_kwargs)

    def fit(self, data, components_):
        raise NotImplementedError

    def transforms(self, W, X):
        S = W.dot(X)
        X_hat = W.T.dot(S)
        return S, X_hat

    def setup_transforms(self, **kwargs):
        """
        Create transform_f and reconstruct_f functions.
        """
        norm_projection = kwargs.get('norm_projection', True)
        X = T.matrix('X')
        W = T.matrix('W')
        if norm_projection:
            epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
            W_norm = T.sqrt(epssumsq)
            Wn = W / W_norm
        else:
            Wn = W
        
        S, X_hat = self.transforms(Wn, X)

        self.transform = theano.function(inputs=[X, W], outputs=[S])
        self.reconstruct = theano.function(inputs=[X, W], outputs=[X_hat])

        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, **kwargs)
        outputs = [loss, error, penalty, mse]
        outputs = [o if o is not None else 0.* loss for o in outputs]

        self.losses_f = theano.function(inputs=[X, W],
                                        outputs=outputs)
    def losses(self, X, W):
        return self.losses_f(X.astype('float32'), W.astype('float32'))
        
    def cost(self, Wn, X, degeneracy=None, lambd=0.,
             p=None, **kwargs):
        """
        Create costs and intermediate values from input variables.
        """
        L, D = Wn.shape
        minC = T.sqrt((L - D) / (D.astype('float32') * (L - 1)))
        S, X_hat = self.transforms(Wn, X)
        gram = T.dot(Wn, Wn.T)
        gram_diff = gram-T.eye(gram.shape[0])
        loss = None
        assert lambd >= 0.

        if degeneracy == 'L2':
            degeneracy = 'Lp'
            p = 2
        elif degeneracy == 'L4':
            degeneracy = 'Lp'
            p = 4

        if degeneracy[0] == 'L':
            try:
                p = int(degeneracy[1:])
                degeneracy = 'Lp'
            except ValueError:
                pass

        if degeneracy == 'RICA':
            error = .5 * T.sum((X_hat-X)**2, axis=0).mean()
        elif degeneracy == 'M1':
            sign = theano.gradient.disconnected_grad(T.switch(gram, 1., -1.))
            error = T.sum(((gram - sign * ((1. - minC) * T.eye(L) +
                minC))**2))
        elif degeneracy == 'M2':
            error = T.sum((gram**2-((1. - minC**2) * T.eye(L) + minC**2))**2)
        elif degeneracy == 'Lp':
            assert isinstance(p, int)
            if (p % 2) == 0:
                error = gram_diff
                for ii in range(p//2):
                    error = error**2
            else:
                error = 1.
                if p > 1:
                    error = gram_diff
                    for ii in range(p//2):
                        error = error**2
                error = error * abs(gram_diff)
            error = (1./p) * T.sum(error)
        elif degeneracy == 'COULOMB':
            epsilon = 0.01
            error = T.sum(1. / T.sqrt(1. + epsilon - gram_diff**2) - 1./np.sqrt(1. + epsilon))
        elif degeneracy == 'COULOMB_F':
            epsilon = 0.01
            error = T.sum((1. / T.sqrt(1. + epsilon - gram_diff**2))
                          - .5 * gram_diff**2/(1.+epsilon)**1.5
                          - 1./np.sqrt(1. + epsilon))
        elif degeneracy == 'RANDOM':
            epsilon = 0.01
            error = - T.sum(T.log(1. + epsilon - gram_diff**2)
                            - np.log(1. + epsilon))
        elif degeneracy == 'RANDOM_F':
            epsilon = 0.01
            error = -T.sum(T.log(1. + epsilon - gram_diff**2)
                           - np.log(1. + epsilon)
                           + gram_diff**2/(1. + epsilon))
        elif degeneracy == 'COHERENCE_SOFT':
            agd = abs(gram_diff)
            agds = agd**2
            gs = gram**2
            boundary = gs.mean()
            boundary = theano.gradient.disconnected_grad(boundary)
            error = T.switch(agds > boundary, agds, 0.).sum()
        elif degeneracy == 'COHERENCE':
            error = abs(gram_diff).max()
        elif degeneracy == 'SM':
            ts = T.tanh(S)
            score = -Wn.T.dot(ts)
            dscore = -(Wn**2).T.dot(1.-ts**2)
            error = ((dscore) + .5*(score**2)).sum(axis=0).mean()
        elif degeneracy is None:
            error = None
        else:
            raise ValueError(degeneracy)

        if ((degeneracy is not None) and
            (lambd == 0. or lambd > 0.) and
            not np.isinf(lambd)):
            loss = error

        if np.isinf(lambd) or degeneracy is None:
            if self.prior == 'soft':
                penalty = T.log(T.cosh(S)).sum(axis=0).mean()
            else:
                penalty = abs(S).sum(axis=0).mean()
            loss = penalty
        elif degeneracy is 'SM':
            assert lambd == 0.
            penalty = T.log(T.cosh(S)).sum(axis=0).mean()
        elif lambd > 0.:
            if self.prior == 'soft':
                penalty = T.log(T.cosh(S)).sum(axis=0).mean()
            else:
                penalty = abs(S).sum(axis=0).mean()
            loss = loss + lambd * penalty
        else:
            penalty = None

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()
        return loss, error, penalty, mse, S, X_hat


class LBFGSB(Optimizer):
    def setup(self, n_sources, n_mixtures, degeneracy, lambd, p,
              **kwargs):
        """
        L-BFGS-B Optimization
        """
        norm_projection = kwargs.get('norm_projection', True)
        X = theano.shared(np.zeros((1, 1), dtype='float32'))
        self.X = X
        Wv = T.vector('W')
        W  = T.reshape(Wv,(n_sources, n_mixtures))
        if norm_projection:
            epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
            W_norm = T.sqrt(epssumsq)
            Wn = W / W_norm
        else:
            Wn = W

        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, degeneracy,
                                                        lambd, p)
        loss_grad = T.grad(loss, Wv)

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()

        self.f_df = theano.function(inputs=[Wv], outputs=[loss,loss_grad])
        if error is None:
            error = 0.*loss
        if penalty is None:
            penalty = 0.*loss
        self.callback_f = theano.function(inputs=[Wv],
                                          outputs=[loss, error, penalty, mse])

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
        w = components_.ravel()
        float_f_df = lambda w: tuple([r.astype('float64') for r in
                                self.f_df(w.astype('float32'))])
        res = minimize(float_f_df, w, jac=True, method='L-BFGS-B', callback=cb)
        w_f = res.x
        l, g = float_f_df(w_f)
        print('ICA with L-BFGS-B done!')
        print('Final loss value: {}'.format(l))
        return w_f.reshape(components_.shape)


class SGD(Optimizer):
    def setup(self, n_sources, n_mixtures, w_0, lambd, degeneracy,
              learning_rule, p, **kwargs):
        """
        SGD optimization
        """
        norm_projection = kwargs.get('norm_projection', True)
        X = T.matrix('X')
        W  = theano.shared(np.random.randn(n_sources, n_mixtures).astype('float32'))
        self.W = W
        if norm_projection:
            epssumsq = T.maximum((W**2).sum(axis=1, keepdims=True), 1e-7)
            W_norm = T.sqrt(epssumsq)
            Wn = W / W_norm
        else:
            Wn = W
        
        loss, error, penalty, mse, S, X_hat = self.cost(Wn, X, degeneracy,
                                                        lambd, p)
        loss_grad = T.grad(loss, W)
        updates = learning_rule([W], [loss_grad])

        X_normed = X/T.sqrt((X**2).sum(axis=0, keepdims=True))
        X_hat_normed = X_hat/T.sqrt((X_hat**2).sum(axis=0, keepdims=True))
        mse = ((X_normed-X_hat_normed)**2).sum(axis=0).mean()
        if error is None:
            error = 0.*loss
        if penalty is None:
            penalty = 0.*loss

        self.train_f = theano.function(inputs=[X],
                                       outputs=[loss, error, penalty, mse],
                                       updates=updates)
        return W, self.train_f

    def fit(self, data, components_,
            tol=1e-4, batch_size=128, n_epochs=1000000,
            patience=10000, seed=20160615):
        """
        Fit components_ from data.
        """
        n_examples = data.shape[1]
        rng = np.random.RandomState(seed)
        self.W.set_value(components_.astype('float32'))
        n_batches = n_examples//batch_size
        if n_batches * batch_size < n_examples:
            n_batches += 1

        lowest_cost = np.inf

        improve = 0

        for ii in range(n_epochs):
            order = rng.permutation(n_examples)
            cur_cost = 0.
            error = 0.
            penalty = 0.
            mse = 0.
            for jj in range(n_batches):
                start = jj * batch_size
                end = (jj + 1) * batch_size
                batch = data[:, order[start:end]].astype('float32')
                res = self.train_f(batch)
                cur_cost += res[0]*batch.shape[1]
                error += res[1]*batch.shape[1]
                penalty += res[2]*batch.shape[1]
                mse += res[3]*batch.shape[1]
            cur_cost /= n_examples
            error /= n_examples
            penalty /= n_examples
            mse /= n_examples
            if self.verbose:
                print('Loss: {}, Error: {}, Penalty: {}, MSE: {}'.format(cur_cost,
                    error, penalty, mse))
            if cur_cost < lowest_cost-tol:
                lowest_cost = cur_cost
                improve = 0
            else:
                improve += 1
            if improve == patience:
                break

        print('ICA with SGD done!' + str(ii))
        print('Final loss value: {}'.format(cur_cost))
        return self.W.get_value()
