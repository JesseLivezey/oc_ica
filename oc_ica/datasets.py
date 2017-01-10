from __future__ import division
import os, h5py
from h5py import File
import numpy as np
from random import sample

from scipy.io import loadmat
from scipy import signal
from sklearn.feature_extraction import image
from skimage.filters import gabor_kernel
from scipy.optimize import minimize

import theano
import theano.tensor as T


def zca(X):
    d, u = np.linalg.eig(np.cov(X))
    M = u.dot(np.diag(np.sqrt(1./d)).dot(u.T))
    X_zca = M.dot(X)
    print u.dot(np.diag(np.sqrt(d)).dot(u.T)).dot(M)
    return X_zca, d, u

def inverse_zca(X, d, u):
    iM = u.dot(np.diag(np.sqrt(d)).dot(u.T))
    return iM.dot(X)

def generate_k_sparse(A, k, n_samples, rng, lambd=1.):
    _, source_dim = A.shape
    #generate sources
    S = rng.laplace(0, lambd, size=(source_dim, n_samples))

    for ii in range(n_samples):
        S[np.argsort(abs(S[:,ii]))[:-k], ii] = 0.

    S /= S.std(axis=-1, keepdims=True)

    #generate data
    X = A.dot(S)

    #preprocess data
    X_mean = X.mean(axis=-1, keepdims=True)
    X -= X_mean
    X_zca = decorrelate(X)
    return X_zca

def generate_imagePatches(datasetPath,patch_size=8,n_patches=200000,\
                            rng=None):

    if rng is None:
            rng = np.random.RandomState(123456)
    parts = ['train','test']
    IMAGES = h5py.File(datasetPath,'r')
    path, name = os.path.split(datasetPath)
    name = name.split('.h5')[0]
    with h5py.File('%s/%s_patches_pca_%i.h5'%(path,name,patch_size),'w') as f:
        for i in xrange(2):
            images = IMAGES['%s'%parts[i]].value
            print images.shape
            max_patches = int(np.ceil(n_patches/images.shape[0]))
            pe = image.PatchExtractor(patch_size=(patch_size, patch_size),
                                    max_patches=max_patches,random_state=rng)
            patches = pe.transform(images)
            print patches.shape
            patches = patches[:n_patches].reshape((n_patches,patches.shape[-1]**2)).T
            if i==0:
                patches_mean = patches.mean(axis=-1, keepdims=True)
                patches -= patches_mean
                u, d, _ = svd(patches, full_matrices=False)
                del _
                K = (u/d).conj().T
                del u,d
                patches = K.dot(patches)
                patches*=np.sqrt(n_patches)
            elif i==1:
                patches -= patches_mean
                patches = K.dot(patches)
                patches*=np.sqrt(n_patches)
            f.create_dataset(name='X_%s'%parts[i],data=patches)
        f.create_dataset(name='X_mean',data=patches_mean)
        f.create_dataset(name='K',data=K)

def whiten(X_train, X_test, whitening='eig'):

    n_samples = X_train.shape[-1]
    X_test_zca = None

    if whitening=='svd':
        u, d, _ = svd(X_train, full_matrices=False)
        del _
        K = (u/d).conj().T
        del u,d
        X_pca = K.dot(X_train)
        X_pca*=np.sqrt(n_samples)
        if X_test is not None:
            X_test_pca = K.dot(X_test)
            X_test_pca*=np.sqrt(n_samples)

    elif whitening=='eig':
        d, u = eig(np.cov(X_train))
        K = np.sqrt(inv(np.diag(d))).dot(u.conj().T)
        X_pca = K.dot(X_train)
        X_zca = u.dot(K).dot(X_train)
        if X_test is not None:
            X_test_pca = K.dot(X_test)
            X_test_zca = u.dot(K).dot(X_test)

    elif whitening is None:
        X_zca = X_train.copy()
        if X_test is not None:
            X_test_zca = X_test.copy()
        
    return X_zca, X_test_zca


def generate_filterbank(freqs=None, n_theta=8, n_freq=8, im_shape=(8, 8),
                        real=True, seed=20160431):
    rng = np.random.RandomState(seed)
    translate_pad = 2
    if real:
        dtype = float
    else:
        dtype = np.complex

    if freqs is None:
        min_freq = 2./min(im_shape)
        freqs = np.linspace(min_freq, .49, n_freq)

    thetas = np.linspace(0, 2. * np.pi, n_theta, endpoint=False)
    if real:
        offsets = [0., np.pi/2.]
    else:
        offsets = [0]
    kernels = np.zeros((len(freqs), len(thetas), len(offsets),
                        im_shape[0]-2*translate_pad,
                        im_shape[1]-2*translate_pad)+im_shape,
                       dtype=dtype)
    for ii, fr in enumerate(freqs):
        for jj, th in enumerate(thetas):
            for kk, of in enumerate(offsets):
                kernel = gabor_kernel(frequency=fr, theta=th,
                                      offset=of)
                if real:
                    kernel = np.real(kernel)
                k_shape = kernel.shape
                for yy in range(translate_pad, im_shape[0]-translate_pad):
                    for xx in range(translate_pad, im_shape[1]-translate_pad):
                        im_y_start = max(0, yy-k_shape[0]//2)
                        im_y_stop = min(im_shape[0], 1+yy+k_shape[0]//2)
                        im_x_start = max(0, xx-k_shape[1]//2)
                        im_x_stop = min(im_shape[1], 1+xx+k_shape[1]//2)

                        ker_y_start = max(0, -yy+k_shape[0]//2)
                        ker_y_stop = min(k_shape[0],
                                1-yy+im_shape[0]-1+k_shape[0]//2)
                        ker_x_start = max(0, -xx+k_shape[1]//2)
                        ker_x_stop = min(k_shape[1],
                                1-xx+im_shape[1]-1+k_shape[1]//2)
                        kernels[ii, jj, kk, xx-translate_pad, yy-translate_pad,
                                im_y_start:im_y_stop,
                                im_x_start:im_x_stop] = kernel[ker_y_start:ker_y_stop,
                                                               ker_x_start:ker_x_stop]
    kernels = kernels.reshape(-1, np.prod(im_shape))
    return kernels

def generate_data(n_sources=None,n_mixtures=64,n_samples=16000,demo_n=1,\
                  rng=None, test_samples=None, **kwargs):
    """
    Data generation function.

    Parameters
    ----------
    n_sources : int
        Number of sources, None defaults to n_mixtures.
    n_mixtures : int
        Number of mixtures.
    n_samples : int
        Number of data samples.
    demo_n : int
        Which type of data.
    rng : np.random.RandomState
        
    """
    if rng==None:
        print '\nRandom seed!'
        rng = np.random.RandomState(np.random.randint(100000))

    if n_sources==None:
        n_sources = 64

    im_size = int(np.sqrt(n_mixtures))

    total_samples = n_samples
    if test_samples is not None:
        total_samples += test_samples

    if demo_n==0:
        kernels = generate_filterbank(im_shape=(im_size, im_size), real=False)
        kernels = kernels[sample(np.arange(kernels.shape[0]),n_sources)]
        A = pinv(kernels)
        r = rng.laplace(0,1e3,size=(n_sources,total_samples))
        for ii in range(total_samples):
            zero = rng.permutation(n_sources)[:9*n_sources//10]
            r[zero, ii] = 0.
        r = r**2*np.sign(r)
        p = rng.uniform(-2*np.pi,2*np.pi,size=(A.shape[1],total_samples))
        S = r*(np.cos(p)+1j*np.sin(p))
        S /= S.std(axis=-1,keepdims=True)
        X = A.dot(S)

    if demo_n==1:
        images = loadmat('IMAGES_RAW.mat')['IMAGESr']
        print images.shape
        im_size = int(np.sqrt(n_mixtures))
        patches = image.PatchExtractor(patch_size=(im_size, im_size),\
                                       max_patches=total_samples//images.shape[-1],
                                       random_state=rng).transform(images.T)
        X = patches.reshape((total_samples,n_mixtures)).T
        if test_samples is not None:
            order = rng.permutation(X.shape[1])
            X_test = X[:, order][:, :test_samples]
            X = np.hstack((X, X_test))
        S = None
        A = None

    if demo_n==2:
        random = kwargs.get('random', False)
        W = rng.randn(n_sources, n_mixtures)
        if not random:
            W_s = T.dvector('W_s')
            W_mat = W_s.reshape((n_sources, n_mixtures))
            W_norms = T.sqrt(T.maximum((W_mat**2).sum(axis=1, keepdims=True),
                                       1e-7))
            W_mat_norm = W_mat/W_norms
            cost = (((W_mat_norm.dot(W_mat_norm.T)-
                      T.eye(n_sources))**2)**2).sum()
            grad = T.grad(cost, W_s).astype('float64')
            f4 = theano.function(inputs=[W_s], outputs=[cost, grad])
            res = minimize(f4, W.ravel(), jac=True, method='L-BFGS-B')
            W = res.x.reshape(n_sources, n_mixtures)
        W /= np.linalg.norm(W, axis=0, keepdims=True)

        A = pinv(W)
        S = rng.laplace(0, 1e3, size=(n_sources, total_samples))

        X = A.dot(S)

    if demo_n==3:
        with h5py.File('van_hateren_downsampled.h5') as f:
            train_images = f['train'].value
            test_images = f['test'].value
        im_size = int(np.sqrt(n_mixtures))
        train_patches = image.PatchExtractor(patch_size=(im_size, im_size),\
                                       max_patches=n_samples//train_images.shape[0],
                                       random_state=rng).transform(train_images)
        X_train = train_patches.reshape(train_patches.shape[0], n_mixtures).T
        if test_samples is not None:
            test_patches = image.PatchExtractor(patch_size=(im_size, im_size),\
                                           max_patches=test_samples//test_images.shape[0],
                                           random_state=rng).transform(test_images)
            X_test = test_patches.reshape(test_patches.shape[0], n_mixtures).T
            X = np.hstack((X_train, X_test))
        else:
            X = X_train
        S = None
        A = None

    if demo_n==4:
        whitening = None
        #with h5py.File('van_hateren_downsampled_patches_pca_8.h5') as f:
        with h5py.File('van_hateren_downsampled_patches_pca_16.h5') as f:
            X_train = f['X_train'].value
            X_test = f['X_test'].value
            K_pre = f['K'].value
        im_size = int(np.sqrt(n_mixtures))

        order = rng.permutation(X_train.shape[1])
        X_train = X_train[:, order][:, :n_samples]
        if test_samples is not None:
            order = rng.permutation(X_test.shape[1])
            X_test = X_test[:, order][:, :test_samples]
            X = np.hstack((X_train, X_test))
        else:
            X = X_train
        S = None
        A = None

    if demo_n==5:        
        filename = '/home/redwood/data/vanhateren/images_curated.h5'
        key = 'van_hateren_good'
        with File(filename,'r') as f:
            images = f[key].value        
        patch_size = 16
        n_dimensions = patch_size**2           
        n_samples = total_samples = 200000                 
        rng = np.random.RandomState(1234)
        patches = image.PatchExtractor(patch_size=(patch_size, patch_size),\
                                       max_patches=total_samples//images.shape[0],
                                       random_state=rng).transform(images)
        X = patches.reshape((patches.shape[0],n_dimensions)).T       
        if test_samples is not None:
            order = rng.permutation(X.shape[1])
            X_test = X[:, order][:, :test_samples]
            X = np.hstack((X, X_test))
        S = None
        A = None

    '''
    data preprocessing
    '''
    
    X_train = X[:, :n_samples]
    X_mean = X_train.mean(axis=-1, keepdims=True)
    X_train -= X_mean
    
    if test_samples is not None:
        X_test = X[:, n_samples:]
        X_test -= X_mean
    else:
        X_test = None

    X_zca, X_test_zca = whiten(X_train, X_test)

    return X_train, X_zca, S, X_mean, A, X_test, X_test_zca
