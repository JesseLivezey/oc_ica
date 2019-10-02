from __future__ import print_function, division
import argparse, h5py, sys, os
import numpy as np

from oc_ica.models import ica, sc
from oc_ica.analysis import evaluate_dgcs
from oc_ica.datasets import generate_k_sparse_analysis


parser = argparse.ArgumentParser(description='Fit models')
parser.add_argument('--n_mixtures', '-n', type=int, default=None)
parser.add_argument('--n_iter', '-i', type=int, default=10)
parser.add_argument('--lambda_min', '-u', type=float, default=-2.)
parser.add_argument('--lambda_max', '-x', type=float, default=2.)
parser.add_argument('--n_lambda', '-b', type=float, default=17)
parser.add_argument('--oc', '-o', type=float, default=None)
parser.add_argument('--priors', '-p', type=str, default=None, nargs='+')
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
parser.add_argument('--k', '-k', type=int, default=None)
parser.add_argument('--w_file', '-w', type=str, default=None)
args = parser.parse_args()

n_mixtures = args.n_mixtures
n_iter = args.n_iter
lambda_min = args.lambda_min
lambda_max = args.lambda_max
n_lambda = args.n_lambda
OC = args.oc
priors = args.priors
models = args.models
k = args.k
w_file = args.w_file

scratch = os.getenv('SCRATCH', '')

print('------------------------------------------')
print('Analysis comparison --> overcompleteness: {}'.format(OC))
print('------------------------------------------')

rng = np.random.RandomState(20191002)

if priors is None:
    W_priors = ['COHERENCE_SOFT']
else:
    W_priors = priors

if models is None:
    models = [2, 4, 'RANDOM', 'RANDOM_F', 'COULOMB_F', 'COULOMB', 'SM']

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

if w_file is None:
    #Create mixing matrices
    n_sources = int(float(OC) * n_mixtures)
    W_array = np.nan * np.ones((len(W_priors), n_iter, n_mixtures, n_sources), dtype='float32')

    for ii, p in enumerate(W_priors):
        print('Generating target angle distributions with prior: {}\n'.format(p))
        for jj in range(n_iter):
            W = np.squeeze(evaluate_dgcs(['random'], [p], n_sources, n_mixtures, rng)[0])
            W_array[ii, jj] = W
    fname = 'W_array-{}_OC-{}_priors-{}.h5'.format(n_mixtures, OC,
                                                   '_'.join(W_priors))
    with h5py.File(os.path.join(scratch, fname), 'w') as f:
        f.create_dataset('W_array', data=W_array)
        f.create_dataset('W_priors', data=np.array([str(p) for p in W_priors]))
else:
    with h5py.File(a_file) as f:
        W_array = f['W_array'].value
        W_priors = f['W_priors'].value
    n_mixtures = W_array.shape[-2]
    n_sources = W_array.shape[-1]
    OC = n_sources / n_mixtures
n_samples = 10 * n_mixtures * n_sources

W_fits = np.nan * np.ones((len(W_priors), len(models), lambdas.size, n_iter) +
                          (n_sources, n_mixtures), dtype='float32')

for ii, p in enumerate(W_priors):
    for jj in range(n_iter):
        W = W_array[ii, jj]
        X = generate_k_sparse_analysis(W, k, n_samples, rng, lambd=0.)
        for kk, model in enumerate(models):
            if model == 'SM':
                lambdas_list = [0.]
            else:
                lambdas_list = lambdas
            for ll, lambd in enumerate(lambdas_list):
                W = fit_ica_model(model, n_sources, lambd, X, rng)
                W_fits[ii, kk, ll, jj] = W

print('\nSaving fits.')
models = [str(m) for m in models]
fname = 'analysis_comparison_mixtures-{}_sources-{}_k-{}_priors-{}_models-{}.h5'.format(n_mixtures,
                                                                               n_sources, k,
                                                                               '_'.join(W_priors),
                                                                               '_'.join(models))
folder = 'analysis_comparison_mixtures-{}_sources-{}_k-{}_priors-{}'.format(n_mixtures, n_sources, k,
                                                                   '_'.join(W_priors))
try:
    os.mkdir(os.path.join(scratch, folder))
except OSError:
    pass

with h5py.File(os.path.join(scratch, folder, fname), 'w') as f:
    f.create_dataset('W_priors', data=np.array([str(p) for p in W_priors]))
    f.create_dataset('models', data=np.array(models))
    f.create_dataset('lambdas', data=lambdas)
    f.create_dataset('W_array', data=W_array)
    f.create_dataset('W_fits', data=W_fits)
