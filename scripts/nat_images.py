from __future__ import division, print_function
import argparse, h5py, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image

from oc_ica.datasets import zca, inverse_zca
from oc_ica.utils import tile_raster_images as tri
from oc_ica.models import sc, ica
from oc_ica.plotting import plot_bases
from oc_ica.sparsity_search import sparsity_search
reload(sc)
reload(ica)


parser = argparse.ArgumentParser(description='Fit models to natural images')
parser.add_argument('--patch_size', '-p', type=int, default=None)
parser.add_argument('--oc', '-o', type=float, default=None)
parser.add_argument('--models', '-m', type=str, default=None, nargs='+')
args = parser.parse_args()

patch_size = args.patch_size
OC = args.oc
models = args.models

scratch = os.getenv('SCRATCH', '')

n_mixtures = patch_size**2           
n_sources = int(float(OC) * n_mixtures)
total_samples = n_mixtures * n_sources
rng = np.random.RandomState(20170110)
if n_mixtures == 64 and n_sources == 128:
    sparsity = 37.
elif n_mixtures == 256 and n_sources == 512:
    sparsity = 150.
else:
    raise ValueError

def fit_ica_model(model, n_sources, X, sparsity, rng, **kwargs):
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
    kwargs['p'] = p
    kwargs['degeneracy'] = model
    kwargs['rng'] = rng
    kwargs['n_sources'] = n_sources
    return sparsity_search(ica.ICA, sparsity, X, **kwargs)


filename = os.path.join(os.environ['SCRATCH'],'data/vanhateren/images_curated.h5')
key = 'van_hateren_good'
with h5py.File(filename,'r') as f:
    images = f[key].value
rng = np.random.RandomState(1234)
patches = image.PatchExtractor(patch_size=(patch_size, patch_size),\
                               max_patches=total_samples//images.shape[0],
                               random_state=rng).transform(images)
X = patches.reshape((patches.shape[0],n_mixtures)).T
X_mean = X.mean(axis=-1, keepdims=True)
X -= X_mean
X_zca, d, u = zca(X)

W_fits = np.full((len(models), n_sources, n_mixtures), np.nan)
lambdas = np.full((len(models),), np.nan)
sparsities = np.full((len(models),), np.nan)

for ii, m in enumerate(models):
    model, lambd, p = fit_ica_model(m, n_sources, X_zca, sparsity, rng)
    W_fits[ii] = model.components_
    lambdas[ii] = lambd
    sparsities[ii] = p

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
