import numpy as np
from models import ica, sc
from analysis import evaluate_dgcs, find_min_k, recovery_statistics
from datasets import generate_k_sparse
import sys
import cPickle


try:
    OC = sys.argv[1]
except:
    OC = 2

try:
    OC = int(OC)
except:
    OC = 2

n_mixtures = 32
global_k = True
n_sources = int(OC) * n_mixtures
n_samples = 5 * n_mixtures * n_sources
rng = np.random.RandomState(20160831)

W_priors = ['L2', 'L4', 'RANDOM', 'COHERENCE']
ica_models = [2, 4, 6, 8, 'RANDOM', 'COULOMB', 'COHERENCE']
lambdas = np.logspace(-2, 2, num=9)
n_iter = 10

# Create mixing matrices
A_dict = dict()
W_dict = dict()

for p in W_priors:
    A_list = []
    W_list = []
    for ii range(n_iter):
        W, _ = evaluate_dgcs(['random'], [p], n_sources, n_mixtures)
        W = np.squeeze(W)
        A_list.append(np.linalg.pinv(W))
        W_list.append(W)
    A_dict[p] = A_list
    W_dict[p] = W_list

if global_k:
    min_k = find_min_k(A_dict)
    assert min_k > 1, 'min_k is too small'

results = np.nan * np.ones((len(W_priors), len(ica_models)+1, lambdas.size, n_iter, 2))
W_fits = np.nan * np.ones((len(W_priors), len(ica_models)+1, lambdas.size, n_iter) +
                          (n_sources, n_mixtures))

W_dict = dict()

for ii, p in enumerate(W_priors):
    W_iter = []
    if not global_k:
        min_k = find_min_k(A_dict[p])
        assert min_k > 1, 'min_k is too small for prior {}'.format(p)
    for jj range(n_iter):
        A = A_dict[p][jj]
        W0 = W_dict[p][jj]
        X = generate_k_sparse(A, k_min, n_samples, rng, lambd=1.)
        for kk, model in enumerate(ica_models):
            for ll, lamb in enumerate(lambdas):
                W = fit_ica_model(model, lambd, X)
                kl, sigma = recovery_statistics(W, W0)
                results[ii, kk, ll, jj] = np.array([kl, sigma])
                W_fits[ii, kk, ll, jj] = W
        for ll, lamb in enumerate(lambdas):
            W = fit_sc_model(lambd, X)
            kl, delta = recovery_statistics(W, W0)
            results[ii, -1, ll, jj] = np.array([kl, delta])
                W_fits[ii, -1, ll, jj] = W

with open('comparison_{}_{}.pkl'.format(n_mixtures, OC), 'w') as f:
    cPickle.dump((A_dict, W_dict, W_fits, results), f)
