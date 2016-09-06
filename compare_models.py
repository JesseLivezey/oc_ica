import numpy as np
from models import ica, sc
from analysis import evaluate_dgcs, find_max_allowed_k, recovery_statistics
from datasets import generate_k_sparse
import sys,pdb
import cPickle


try:
    OC = sys.argv[1]
except:
    OC = 2

try:
    OC = float(OC)
except:
    OC = 2

OC = 2

print '\n------------------------------------------'
print '\nICA-SC comparison --> overcompleteness: %i'%OC
print '\n------------------------------------------'

n_mixtures = 128
global_k = True
n_sources = int(float(OC) * n_mixtures)
n_samples = 5 * n_mixtures * n_sources
rng = np.random.RandomState(20160831)

#W_priors = ['L2', 'L4', 'RANDOM', 'COHERENCE']
W_priors = ['L2', 'L4', 'RANDOM']
ica_models = [2, 4, 6, 8, 'RANDOM', 'COULOMB', 'COHERENCE']
#ica_models = [2, 4]
lambdas = np.logspace(-2, 2, num=9)
#lambdas = np.array([.1,1.],dtype=np.float32)#np.logspace(-1, 2, num=3).astype(np.float32)
#n_iter = 10
n_iter = 10

def fit_ica_model(model, dim_sparse, lambd, X):
    dim_input = X.shape[0]
    if type(model) is int:
        p = model
        model='Lp'
    else:
        p = None
    ica_model = ica.ICA(n_mixtures=dim_input,n_sources=dim_sparse,lambd=lambd,
                degeneracy=model,p=p)
    ica_model.fit(X)
    return ica_model.components_

def fit_sc_model(dim_sparse,lambd, X):
    dim_input = X.shape[0]
    sc_model = sc.SparseCoding(n_mixtures=dim_input,
	                       n_sources=dim_sparse,
                               lambd=lambd)
    sc_model.fit(X)
    return sc_model.components_

#Create mixing matrices
A_dict = dict()
W_dict = dict()

for p in W_priors:
    '\n - Generating target angle distributions with prior: [%s]'%p
    A_list = []
    W_list = []
    for ii in range(n_iter):
        AT = np.squeeze(evaluate_dgcs(['random'], [p], n_sources, n_mixtures)[0])
        A = AT.T
        A_list.append(A)
        W_list.append(np.linalg.pinv(A))
    A_dict[p] = A_list
    W_dict[p] = W_list

if global_k:
    min_k =  find_max_allowed_k(A_dict, n_sources)
    print  '\nGlobal min. k-value: %i'%min_k
    assert min_k > 1, 'min_k is too small'


results = np.nan * np.ones((len(W_priors), len(ica_models)+1, lambdas.size, n_iter, 2))
W_fits = np.nan * np.ones((len(W_priors), len(ica_models)+1, lambdas.size, n_iter) +
                          (n_sources, n_mixtures))
min_ks = np.nan * np.ones(len(W_priors))

for ii, p in enumerate(W_priors):
    W_iter = []
    if not global_k:
        min_k = find_max_allowed_k(A_dict[p], n_sources)
        print  '\nLocal min k-value: %i'%min_k
        assert min_k > 1, 'min_k is too small for prior {}'.format(p)
    min_ks.append(min_k)
    for jj in range(n_iter):
        A = A_dict[p][jj]
        W0 = W_dict[p][jj]
        X = generate_k_sparse(A, min_k, n_samples, rng, lambd=1.)
        for kk, model in enumerate(ica_models):
            for ll, lambd in enumerate(lambdas):
                W = fit_ica_model(model, n_sources, lambd, X)
                kl, sigma = recovery_statistics(W, W0)
                results[ii, kk, ll, jj] = np.array([kl, sigma])
                W_fits[ii, kk, ll, jj] = W
        for ll, lambd in enumerate(lambdas):
            W = fit_sc_model(n_sources, lambd, X)
            kl, delta = recovery_statistics(W, W0)
            results[ii, -1, ll, jj] = np.array([kl, delta])
            W_fits[ii, -1, ll, jj] = W

with open('comparison_{}_{}.pkl'.format(n_mixtures, OC), 'w') as f:
    cPickle.dump((A_dict, W_dict, W_fits, min_ks, results), f)
