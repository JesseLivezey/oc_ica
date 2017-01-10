from __future__ import division, print_function
import h5py, os
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


patch_size = 16
overcompleteness = 2
degeneracy= 'Lp'
p=2
lambd = 10.

n_mixtures = patch_size**2           
n_sources = n_mixtures * overcompleteness
total_samples = n_mixtures * n_sources

filename = os.path.join(os.environ['HOME'],'Development/data/vanhateren/images_curated.h5')
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

Xp = inverse_zca(X, d, u)
print(np.allclose(X, Xp))
print(X-Xp)

print('creating model')
model, lambd, p = sparsity_search(ica.ICA, 150., X_zca,
        degeneracy=degeneracy, p=p, n_sources=n_sources)
print(lambd)
"""
model = ica.ICA(n_mixtures=n_mixtures,n_sources=n_sources,lambd=lambd,
                degeneracy=degeneracy_control,p=p)

print('fitting')
model.fit(X_zca)
"""
print(model.losses(X_zca))
bases = model.components_
plot_bases(bases)
bases = inverse_zca(bases.T, d, u).T
plot_bases(bases)
