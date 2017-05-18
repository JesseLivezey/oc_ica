from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict


cmap = plt.cm.viridis

models = ['QUASI-ORTHO', '2', '4', 'COHERENCE_SOFT', 'COULOMB', 'COULOMB_F', 'RANDOM', 'RANDOM_F', 'SM', 'SC']

n_ica = 6
ica_colors = np.linspace(1, 0, n_ica-1)
ica_colors = ['gray'] + [cmap(ica_colors[-1]), cmap(ica_colors[0]), cmap(ica_colors[1]),
                          cmap(ica_colors[2]), cmap(ica_colors[2]),
                          cmap(ica_colors[3]), cmap(ica_colors[3])]

model_colors = ica_colors + ['cyan', 'magenta']
colors = {m: c for m, c in zip(models, model_colors)}
colors['INIT'] = 'red'

model_line_styles = ['-', '-', '-', '-', '-', ':', '-', ':', '-', '-']
line_styles = {m: s for m, s in zip(models, model_line_styles)}
line_styles['INIT'] = '--'

model_labels = ['Quasi-ortho', r'$L_2$', r'$L_4$', 'Soft Coher.', 'Coulomb',
                'Flat Coulomb', 'Rand. Prior', 'Flat Rand. Prior',
                'Score Matching', 'Sparse Coding']
labels = {m: l for m, l in zip(models, model_labels)}
labels['INIT'] = 'Init.'

legend_fontsize = 8
label_fontsize = 8
ticklabel_fontsize = 8
letter_fontsize=12
lw = 1
angle_bins = np.linspace(0, 90, 181)
ds = 'steps-pre'
