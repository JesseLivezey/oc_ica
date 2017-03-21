from __future__ import division, print_function
import numpy as np


def sparsity_search(model_c, sparsity, X, log_lambd_range=None,
                    n_iter=20, tol=1e-2, **model_kwargs):
    """
    Does a binary search in log_lambd_range to match sparsity.
    Expects sparsity to be a monotonically decreasing function of
    log_lambd. May not behave well if this is not the case.

    Parameters
    ----------
    model_c : ICA constructor
        Model class constructor.
    sparsity : float
        Target sparsity.
    X : ndarray
        Data array, (n_mixtures, n_samples)
    log_lambd_range : list
        Lower and upper bound for log_lambd search.
    n_iter : int
        Number of binary search steps within range.
    tol : float
        Early stopping tolerance.
    model_kwargs : dict
        kwargs to be passed to ICA constructor.

    Returns
    -------
    model : ICA instance
        Model trained with lambd which gives closest match to sparsity.
    lambd : float
        Lambd which gives closest match to sparsity.
    p : float
        Final sparsity of model.
    """

    if log_lambd_range is None:
        log_lambd_range = [-4, 6]

    n_mixtures = X.shape[0]

    lower = log_lambd_range[0]
    upper = log_lambd_range[1]

    lower_model = model_c(n_mixtures, lambd=np.power(10., lower), **model_kwargs)
    lower_model.fit(X)
    l, e, p, m = lower_model.losses(X)
    if sparsity >= p-tol:
        return lower_model, np.power(10., lower), p

    upper_model = model_c(n_mixtures, lambd=np.power(10., upper), **model_kwargs)
    upper_model.fit(X)
    l, e, p, m = upper_model.losses(X)
    if sparsity <= p+tol:
        return upper_model, np.power(10., upper), p

    for ii in range(n_iter):
        mid = (lower+upper)/2.
        lambd = np.power(10., mid)
        model = model_c(n_mixtures, lambd=lambd, **model_kwargs)
        model.fit(X)
        l, e, p, m = model.losses(X)
        if abs(sparsity - p) < tol:
            return model, lambd, p
        elif p < sparsity:
            upper = mid
        else:
            lower = mid
    return model, lambd, p
