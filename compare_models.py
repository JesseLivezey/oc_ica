from __future__ import print_function, division
import h5py, sys
import numpy as np

from models import ica, sc
from optimizers.ica_optimizers import sgd
from analysis import evaluate_dgcs, find_max_allowed_k
from datasets import generate_k_sparse


try:
    OC = sys.argv[1]
except:
    OC = 2

try:
    OC = float(OC)
except:
    OC = 2

OC = 2

print('------------------------------------------')
print('ICA-SC comparison --> overcompleteness: {}'.format(OC))
print('------------------------------------------')

n_mixtures = 128
global_k = True
n_sources = int(float(OC) * n_mixtures)
n_samples = 5 * n_mixtures * n_sources
rng = np.random.RandomState(20160831)

A_priors = ['L2', 'L4', 'RANDOM', 'COHERENCE']
ica_models = [2, 4, 6, 'COHERENCE', 'RANDOM', 'RANDOM_F', 'COULOMB_F', 'COULOMB']
model_kwargs = [dict(), dict(), dict(), {'optimizer': 'sgd', 'learning_rule': sgd},
                dict(), dict(), dict(), dict()]
lambdas = np.logspace(-2, 2, num=17)
n_iter = 40

def fit_ica_model(model, dim_sparse, lambd, X, rng, **kwargs):
    dim_input = X.shape[0]
    if isinstance(model, int):
        p = model
        model='Lp'
    else:
        p = None
    ica_model = ica.ICA(n_mixtures=dim_input, n_sources=dim_sparse,
                        lambd=lambd, degeneracy=model, p=p, rng=rng, **kwargs)
    ica_model.fit(X)
    return ica_model.components_

def fit_sc_model(dim_sparse, lambd, X, rng, **kwargs):
    dim_input = X.shape[0]
    sc_model = sc.SparseCoding(n_mixtures=dim_input,
                               n_sources=dim_sparse,
                               lambd=lambd, rng=rng, **kwargs)
    sc_model.fit(X)
    return sc_model.components_

#Create mixing matrices
A_array = np.nan * np.ones((len(A_priors), n_iter, n_mixtures, n_sources))

for ii, p in enumerate(A_priors):
    print('Generating target angle distributions with prior: {}'.format(p))
    for jj in range(n_iter):
        AT = np.squeeze(evaluate_dgcs(['random'], [p], n_sources, n_mixtures, rng)[0])
        A_array[ii, jj] = AT.T

if global_k:
    min_k =  find_max_allowed_k(A_array)
    print('Global min. k-value: {}'.format(min_k))
    assert min_k > 1, 'min_k is too small'


W_fits = np.nan * np.ones((len(A_priors), len(ica_models)+1, lambdas.size, n_iter) +
                          (n_sources, n_mixtures))
min_ks = np.nan * np.ones(len(A_priors))

for ii, p in enumerate(A_priors):
    if not global_k:
        min_k = find_max_allowed_k(A_array[ii])
        print('Local min k-value: {}'.format(min_k))
        assert min_k > 1, 'min_k is too small for prior {}'.format(p)
    min_ks[ii] = min_k
    for jj in range(n_iter):
        A = A_array[ii, jj]
        X = generate_k_sparse(A, min_k, n_samples, rng, lambd=1.)
        for kk, model in enumerate(ica_models):
            for ll, lambd in enumerate(lambdas):
                W = fit_ica_model(model, n_sources, lambd, X, rng)
                W_fits[ii, kk, ll, jj] = W
        for ll, lambd in enumerate(lambdas):
            W = fit_sc_model(n_sources, lambd, X, rng)
            W_fits[ii, -1, ll, jj] = W

with h5py.File('comparison_{}_{}_{}.h5'.format(n_mixtures, OC, '_'.join(A_priors)), 'w') as f:
    f.create_dataset('A_priors', data=np.array([str(p) for p in A_priors]))
    f.create_dataset('ica_models', data=np.array(ica_models))
    f.create_dataset('lambdas', data=lambdas)
    f.create_dataset('A_array', data=A_array)
    f.create_dataset('W_fits', data=W_fits)
    f.create_dataset('min_ks', data=np.array(min_ks))
