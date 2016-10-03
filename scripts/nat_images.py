from datasets import generate_data as gd
import numpy as np
import matplotlib.pyplot as plt
from utils import tile_raster_images as tri
from h5py import File
from sklearn.feature_extraction import image

from oc_ica.models import sc, ica
reload(sc)
reload(ica)

filename = '/home/jesse/Development/data/vanhateren/images_curated.h5'
key = 'van_hateren_good'
with File(filename,'r') as f:
    images = f[key].value
patch_size = 16
n_dimensions = patch_size**2           
total_samples = 200000                 
rng = np.random.RandomState(1234)
patches = image.PatchExtractor(patch_size=(patch_size, patch_size),\
                               max_patches=total_samples//images.shape[0],
                               random_state=rng).transform(images)
X = patches.reshape((patches.shape[0],n_dimensions)).T
X_mean = X.mean(axis=-1, keepdims=True)
X -= X_mean
def decorrelate(X):
    d, u = np.linalg.eig(np.cov(X))
    K = np.sqrt(np.linalg.inv(np.diag(d))).dot(u.T)
    X_zca = u.dot(K).dot(X)
    return X_zca

#X = gd()[0]
X_zca = decorrelate(X)

overcompleteness = 2
"""
degeneracy_control = 'Lp'
p=4
"""
lambd = .1
n_mixtures = X_zca.shape[0]
n_sources  = n_mixtures*overcompleteness
#ica = ocica.ICA(n_mixtures=n_mixtures,n_sources=n_sources,lambd=lambd,
#                degeneracy=degeneracy_control,optimizer='sgd',learning_rule=optimizers.adam,p=p)
"""
ica = ocica.ICA(n_mixtures=n_mixtures,n_sources=n_sources,lambd=lambd,
                degeneracy=degeneracy_control,p=p)

ica.fit(X_zca)
"""
model = sc.SparseCoding(n_mixtures, n_sources, lambd)
model.fit(X_zca)


def plot_bases(bases,figsize=None):
    """Plots a basis set
    
    Parameters:
    ----------
    bases : ndarray
           Set of basis.
           Dimension: n_costs X n_vectors X n_dims
    figname: string, optional
           Name of the figure
    """
    n_pixels = int(np.sqrt(bases.shape[1]))
    n_bases  = int(np.sqrt(bases.shape[0]))
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)
    fig.clf()
    ax = plt.axes()
    im = tri(bases,(n_pixels,n_pixels),(n_bases,n_bases),
                (2,2), scale_rows_to_unit_interval=False,
                output_pixel_vals=False)
    ax.imshow(im, interpolation='nearest', aspect='auto', cmap='gray')
    ax.set_axis_off()
    plt.show()

bases = model.components_
plot_bases(bases,figsize=(10,10))
