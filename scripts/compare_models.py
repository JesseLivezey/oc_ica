from __future__ import print_function, division
import argparse, h5py, sys, os
import numpy as np

from oc_ica.models import ica, sc
from oc_ica.optimizers.ica_optimizers import sgd
from oc_ica.analysis import evaluate_dgcs, find_max_allowed_k
from oc_ica.datasets import generate_k_sparse


parser = argparse.ArgumentParser(description='Fit models')
parser.add_argument('--n_mixtures', '-n', type=int, default=None)
parser.add_argument('--n_iter', '-i', type=int, default=10)
parser.add_argument('--lambda_min', '-u', type=float, default=-2.)
parser.add_argument('--lambda_max', '-x', type=float, default=2.)
parser.add_argument('--n_lambda', '-b', type=float, default=17)
parser.add_argument('--oc', '-o', type=float, default=None)
parser.add_argument('--priors', '-p', type=str, default=None, nargs='+')
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
parser.add_argument('--a', '-a', type=str, default=None)
parser.add_argument('--k', '-k', type=int, default=None)
parser.add_argument('--generate', '-g', default=False, action='store_true')
parser.add_argument('--sgd_COHERENCE', '-s', default=False, action='store_true')
parser.add_argument('--keep_max', '-x', default=False, action='store_true')
args = parser.parse_args()

n_mixtures = args.n_mixtures
OC = args.oc
priors = args.priors
models = args.models
a_file = args.a
generate = args.generate
k = args.k
n_iter = args.n_iter
lambda_min = args.lambda_min
lambda_max = args.lambda_max
n_lambda = args.n_lambda
sgd_COHERENCE = args.sgd_COHERENCE
keep_max = args.keep_max

if sgd_COHERENCE:
    assert generate

scratch = os.getenv('SCRATCH', '')

print('------------------------------------------')
print('ICA-SC comparison --> overcompleteness: {}'.format(OC))
print('------------------------------------------')

global_k = True
rng = np.random.RandomState(20160831)

if priors is None:
    A_priors = ['L2', 'L4', 'RANDOM', 'COHERENCE']
else:
    A_priors = priors

if models is None:
    models = [2, 4, 6, 'COHERENCE', 'RANDOM', 'RANDOM_F', 'COULOMB_F', 'COULOMB', 'SC']

lambdas = np.logspace(lambda_min, lambda_max, num=n_lambda)

def fit_ica_model(model, dim_sparse, lambd, X, rng, **kwargs):
    dim_input = X.shape[0]
    try:
        model = int(model)
    except ValueError:
        pass
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
    n_sources = int(float(OC) * n_mixtures)
    A_array = np.nan * np.ones((len(A_priors), n_iter, n_mixtures, n_sources), dtype='float32')

    for ii, p in enumerate(A_priors):
        if sgd_COHERENCE and p == 'COHERENCE':
            kwargs = {'optimizer': 'sgd',
                      'learning_rule': sgd}
        else:
            kwargs = {}
        print('Generating target angle distributions with prior: {}\n'.format(p))
        for jj in range(n_iter):
            AT = np.squeeze(evaluate_dgcs(['random'], [p], n_sources, n_mixtures, rng, **kwargs)[0])
            A_array[ii, jj] = AT.T
    fname = 'a_array-{}_OC-{}_priors-{}.h5'.format(n_mixtures, OC,
                                                   '_'.join(A_priors))
    with h5py.File(os.path.join(scratch, fname), 'w') as f:
        f.create_dataset('A_array', data=A_array)
        f.create_dataset('A_priors', data=np.array([str(p) for p in A_priors]))
else:
    with h5py.File(a_file) as f:
        A_array = f['A_array'].value
        A_priors = f['A_priors'].value
    n_mixtures = A_array.shape[-2]
    n_sources = A_array.shape[-1]
    OC = n_sources / n_mixtures
n_samples = 10 * n_mixtures * n_sources

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
        X = generate_k_sparse(A, min_k, n_samples, rng, lambd=1.,
                              keep_max=keep_max)
        for kk, model in enumerate(models):
            if model == 'SM':
                lambdas_list = [0.]
            else:
                lambdas_list = lambdas
            for ll, lambd in enumerate(lambdas_list):
                if isinstance(model, str) and model[:2] == 'SC':
                    if 'SOFT' in model:
                        kwargs = {'prior': 'soft'}
                    else:
                        kwargs = {'prior': 'hard'}
                    W = fit_sc_model(n_sources, lambd, X, rng, **kwargs)
                    W_fits[ii, kk, ll, jj] = W
                else:
                    W = fit_ica_model(model, n_sources, lambd, X, rng)
                    W_fits[ii, kk, ll, jj] = W

print('\nSaving fits.')
models = [str(m) for m in models]
fname = 'comparison_mixtures-{}_sources-{}_k-{}_priors-{}_models-{}_keep_max-{}.h5'.format(n_mixtures,
                                                                               n_sources, k,
                                                                               '_'.join(A_priors),
                                                                               '_'.join(models),
                                                                               keep_max)
folder = 'comparison_mixtures-{}_sources-{}_k-{}_priors-{}_keep_max-{}'.format(n_mixtures, n_sources, k,
                                                                   '_'.join(A_priors),
                                                                   keep_max)
try:
    os.mkdir(os.path.join(scratch, folder))
except OSError:
    pass

with h5py.File(os.path.join(scratch, folder, fname), 'w') as f:
    f.create_dataset('A_priors', data=np.array([str(p) for p in A_priors]))
    f.create_dataset('models', data=np.array(models))
    f.create_dataset('lambdas', data=lambdas)
    f.create_dataset('A_array', data=A_array)
    f.create_dataset('W_fits', data=W_fits)
    f.create_dataset('min_ks', data=np.array(min_ks))
