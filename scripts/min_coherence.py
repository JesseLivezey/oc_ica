from __future__ import print_function, division
import argparse, h5py, sys, os
import numpy as np

from oc_ica.models import ica
from oc_ica.analysis import get_W, normalize_W


parser = argparse.ArgumentParser(description='Fit models')
parser.add_argument('--n_mixtures', '-n', type=int, default=None)
parser.add_argument('--n_iter', '-i', type=int, default=10)
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
parser.add_argument('--ocs', '-o', type=int, default=None, nargs='+')
args = parser.parse_args()

n_mixtures = args.n_mixtures
models = args.models
n_iter = args.n_iter
ocs = args.ocs

scratch = os.getenv('SCRATCH', '')

rng = np.random.RandomState(20170518)

if models is None:
    models = [2, 4, 'RANDOM', 'RANDOM_F', 'COULOMB', 'COULOMB_F']

if ocs is None:
    ocs = [1., 1.125, 1.25, 1.5, 1.75, 2., 2.5, 3.]

n_sources = int(float(sorted(ocs)[-1]) * n_mixtures)

W_fits = np.nan * np.ones((len(models), len(ocs), n_iter) +
                          (n_sources, n_mixtures), dtype='float32')
W_orig = np.nan * np.ones((len(ocs), n_iter) +
                          (n_sources, n_mixtures), dtype='float32')

for kk, OC in enumerate(ocs):
    n_sources = int(float(OC) * n_mixtures)
    for ii in range(n_iter):
        W = normalize_W(rng.randn(n_sources, n_mixtures))
        W_orig[kk, ii, :n_sources] = W
        for jj, model in enumerate(models):
            if isinstance(model, int):
                model = 'L{}'.format(model)
            W_fits[jj, kk, ii, :n_sources] = get_W(W, model, rng)

print('\nSaving fits.')
models = [str(m) for m in models]
ocs = [str(oc) for oc in ocs]
fname = 'data_free-{}_sources-{}_ocs-{}_models-{}.h5'.format(n_mixtures,
                                                      n_sources,
                                                      '_'.join(ocs),
                                                      '_'.join(models))
folder = 'data_free_sources-{}'.format(n_mixtures)

try:
    os.mkdir(os.path.join(scratch, folder))
except OSError:
    pass

with h5py.File(os.path.join(scratch, folder, fname), 'w') as f:
    f.create_dataset('models', data=np.array(models))
    f.create_dataset('W_fits', data=W_fits)
    f.create_dataset('W_orig', data=W_orig)
    f.create_dataset('ocs', data=np.array(ocs))
