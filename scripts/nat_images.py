from __future__ import division, print_function
import argparse, h5py, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image

from oc_ica.datasets import zca
from oc_ica.utils import tile_raster_images as tri
from oc_ica.models import sc, ica
from oc_ica.plotting import plot_bases
from oc_ica.sparsity_search import sparsity_search
reload(sc)
reload(ica)


parser = argparse.ArgumentParser(description='Fit models to natural images')
parser.add_argument('--patch_size', '-p', type=int, default=None)
parser.add_argument('--n_iter', '-n', type=int, default=10)
parser.add_argument('--oc', '-o', type=float, default=None)
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
args = parser.parse_args()

patch_size = args.patch_size
OC = args.oc
models = args.models
n_iter = args.n_iter

scratch = os.getenv('SCRATCH', '')

n_mixtures = patch_size**2           
n_sources = int(float(OC) * n_mixtures)
total_samples = 10 * n_mixtures * n_sources
rng = np.random.RandomState(20170110)
if n_mixtures == 64 and n_sources == 128:
    sparsity = 38.
elif n_mixtures == 256 and n_sources == 512:
    sparsity = 150.
else:
    raise ValueError

def fit_ica_model(model, n_sources, X, sparsity, rng, **kwargs):
    p = None
    try:
        model = int(model)
        p = model
        model='Lp'
    except ValueError:
        if model == 'SM':
            n_mixtures = X.shape[0]
            model = ica.ICA(n_mixtures, n_sources, degeneracy='SM',
                            lambd=0.)
            model.fit(X)
            sparsity = model.losses(X)[2]
            return model, None, sparsity
    kwargs['p'] = p
    kwargs['degeneracy'] = model
    kwargs['rng'] = rng
    kwargs['n_sources'] = n_sources
    return sparsity_search(ica.ICA, sparsity, X, **kwargs)


filename = os.path.join(os.environ['SCRATCH'],'data/vanhateren/images_curated.h5')
key = 'van_hateren_good'
with h5py.File(filename,'r') as f:
    images = f[key].value
patches = image.PatchExtractor(patch_size=(patch_size, patch_size),
                               max_patches=total_samples//images.shape[0],
                               random_state=rng).transform(images)
X = patches.reshape((patches.shape[0],n_mixtures)).T
X_mean = X.mean(axis=-1, keepdims=True)
X -= X_mean
X_zca, d, u = zca(X)

W_fits = np.full((len(models), n_iter, n_sources, n_mixtures), np.nan)
lambdas = np.full((len(models), n_iter), np.nan)
sparsities = np.full((len(models), n_iter), np.nan)

for ii, m in enumerate(models):
    for jj in range(n_iter):
        model, lambd, p = fit_ica_model(m, n_sources, X_zca, sparsity, rng)
        W_fits[ii, jj] = model.components_
        lambdas[ii, jj] = lambd
        sparsities[ii, jj] = p

fname = 'nat_images_mixtures-{}_sources-{}_models-{}.h5'.format(n_mixtures,
                                                                n_sources,
                                                                '_'.join(models))
folder = 'nat_images_mixtures-{}_sources-{}'.format(n_mixtures, n_sources)

try:
    os.mkdir(os.path.join(scratch, folder))
except OSError:
    pass

with h5py.File(os.path.join(scratch, folder, fname), 'w') as f:
    f.create_dataset('W_fits', data=W_fits)
    f.create_dataset('lambdas', data=lambdas)
    f.create_dataset('sparsities', data=sparsities)
