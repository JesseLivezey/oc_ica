
# coding: utf-8

# In[6]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[14]:

def compute_angles(w):
    w = w/np.linalg.norm(w, axis=-1, keepdims=True)
    gram = w.dot(w.T)
    gram_off_diag = gram[np.tri(gram.shape[0], k=-1, dtype=bool)]
    return np.arccos(abs(gram_off_diag))/np.pi*180

def gram(w):
    w = w/np.linalg.norm(w, axis=-1, keepdims=True)
    return w.dot(w.T)


# In[36]:

w = np.random.randn(64*25, 64)
g = gram(w)
g = g - np.eye(g.shape[0])

hists = []
angles = compute_angles(w)
vals, bins = np.histogram(angles, bins=np.arange(0, 91))
hists.append(vals)
print(angles.min())
plt.figure()
for ii in range(7):
    for jj in range(200):
        g_max = g.max(axis=1)
        arg_max = np.argmax(g_max)
        w = np.delete(w, arg_max, axis=0)
        g = np.delete(g, arg_max, axis=0)
        g = np.delete(g, arg_max, axis=1)
    angles = compute_angles(w)
    print(angles.min())
    vals, bins = np.histogram(angles, bins=np.arange(0, 91))
    hists.append(vals)


# In[42]:

for ii in range(len(hists)-1):
    plt.plot(bins[:-1], (hists[0]-hists[ii+1])/(hists[0]+1e-3), drawstyle='steps-mid', label=str(ii+1))
#plt.yscale('log')
plt.legend(loc='center left')
plt.xlim([0, 90])


# In[45]:

for ii in range(len(hists)-1):
    plt.plot(bins[:-1], (hists[ii]-hists[ii+1]), drawstyle='steps-mid', label=str(ii+1))
#plt.yscale('log')
plt.legend(loc='center left')
plt.xlim([0, 90])

