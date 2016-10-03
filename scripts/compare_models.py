from __future__ import print_function, division
import argparse, h5py, sys
import numpy as np

from oc_ica.models import ica, sc
from oc_ica.optimizers.ica_optimizers import sgd
from oc_ica.analysis import evaluate_dgcs, find_max_allowed_k
from oc_ica.datasets import generate_k_sparse


parser = argparse.ArgumentParser(description='Fit models')
parser.add_argument('--n_mixtures', '-n', type=int, default=128)
parser.add_argument('--oc', '-o', type=float, default=2.)
parser.add_argument('--priors', '-p', type=str, default=None, nargs='+')
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
parser.add_argument('--a', '-a', type=str, default=None)
parser.add_argument('--k', '-k', type=int, default=None)
parser.add_argument('--generate', '-g', default=False, action='store_true')
args = parser.parse_args()

n_mixtures = args.n_mixtures
OC = args.oc
priors = args.priors
models = args.models
a_file = args.a
generate = args.generate
k = args.k

print('------------------------------------------')
print('ICA-SC comparison --> overcompleteness: {}'.format(OC))
print('------------------------------------------')

global_k = True
n_sources = int(float(OC) * n_mixtures)
n_samples = 5 * n_mixtures * n_sources
rng = np.random.RandomState(20160831)

if priors is None:
    A_priors = ['L2', 'L4', 'RANDOM', 'COHERENCE']
else:
    A_priors = priors

if models is None:
    models = [2, 4, 6, 'COHERENCE', 'RANDOM', 'RANDOM_F', 'COULOMB_F', 'COULOMB', 'SC']

model_kwargs = {'COHERENCE': {'optimizer': 'sgd', 'learning_rule': sgd}}

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

if a_file is None:
    #Create mixing matrices
    A_array = np.nan * np.ones((len(A_priors), n_iter, n_mixtures, n_sources), dtype='float32')

    for ii, p in enumerate(A_priors):
        print('Generating target angle distributions with prior: {}\n'.format(p))
        kwargs = model_kwargs.get(p, dict())
        for jj in range(n_iter):
            AT = np.squeeze(evaluate_dgcs(['random'], [p], n_sources, n_mixtures, rng, **kwargs)[0])
            A_array[ii, jj] = AT.T
    with h5py.File('a_array-{}_OC-{}_priors-{}.h5'.format(n_mixtures, OC,
                                                          '_'.join(A_priors)), 'w') as f:
        f.create_dataset('A_array', data=A_array)
        f.create_dataset('A_priors', data=np.array([str(p) for p in A_priors]))
else:
    with h5py.File(a_file) as f:
        A_array = f['A_array'].value
        A_priors = f['A_priors'].value

if generate:
    sys.exit(0)

if global_k and k is None:
    min_k =  find_max_allowed_k(A_array)
    print('Global min. k-value: {}'.format(min_k))
    assert min_k > 1, 'min_k is too small'


W_fits = np.nan * np.ones((len(A_priors), len(models), lambdas.size, n_iter) +
                          (n_sources, n_mixtures), dtype='float32')
min_ks = np.nan * np.ones(len(A_priors))

for ii, p in enumerate(A_priors):
    if not global_k and k is None:
        min_k = find_max_allowed_k(A_array[ii])
        print('Local min k-value: {}'.format(min_k))
        assert min_k > 1, 'min_k is too small for prior {}'.format(p)
    if k is not None:
        min_k = k
    min_ks[ii] = min_k
    for jj in range(n_iter):
        A = A_array[ii, jj]
        X = generate_k_sparse(A, min_k, n_samples, rng, lambd=1.)
        for kk, model in enumerate(models):
            for ll, lambd in enumerate(lambdas):
                if model == 'SC':
                    kwargs = model_kwargs.get(model, dict())
                    W = fit_sc_model(n_sources, lambd, X, rng, **kwargs)
                    W_fits[ii, kk, ll, jj] = W
                else:
                    try:
                        model = int(model)
                    except ValueError:
                        pass
                    kwargs = model_kwargs.get(model, dict())
                    W = fit_ica_model(model, n_sources, lambd, X, rng, **kwargs)
                    W_fits[ii, kk, ll, jj] = W
                print(kwargs)

print('\nSaving fits.')
models = [str(m) for m in models]
with h5py.File('comparison_mixtures-{}_OC-{}_priors-{}_models-{}.h5'.format(n_mixtures, OC,
                                                                            '_'.join(A_priors), '_'.join(models)), 'w') as f:
    f.create_dataset('A_priors', data=np.array([str(p) for p in A_priors]))
    f.create_dataset('models', data=np.array(models))
    f.create_dataset('lambdas', data=lambdas)
    f.create_dataset('A_array', data=A_array)
    f.create_dataset('W_fits', data=W_fits)
    f.create_dataset('min_ks', data=np.array(min_ks))
