import argparse
import numpy as np
import os
import pdb


base_dirs = ['results1', 'results2', 'results3', 'results4', 'results5']  # non-MOM
base_dirs = ['results6', 'results7', 'results8', 'results9', 'results10']  # only-MOM
base_dirs = ['results11', 'results12', 'results13', 'results14', 'results15']  # MOMbigbatch

# All models together.
log_dirs = ['iwgan_2iw', 'iwgan_2mom', 'iwgan_2sn',
            'mmdgan_2iw', 'mmdgan_2mom', 'mmdgan_2sn',
            'cgan_2d', 'upsample_2d',

            'iwgan_4iw', 'iwgan_4mom', 'iwgan_4sn',
            'mmdgan_4iw', 'mmdgan_4mom', 'mmdgan_4sn',
            'cgan_4d', 'upsample_4d', 

            'iwgan_10iw', 'iwgan_10mom', 'iwgan_10sn',
            'mmdgan_10iw', 'mmdgan_10mom', 'mmdgan_10sn',
             'cgan_10d', 'upsample_10d']
# Just mmdgan models.
log_dirs = ['mmdgan_2iw', 'mmdgan_2mom', 'mmdgan_2sn',
            'mmdgan_4iw', 'mmdgan_4mom', 'mmdgan_4sn',
            'mmdgan_10iw', 'mmdgan_10mom', 'mmdgan_10sn']
# Just mom models.
log_dirs = ['iwgan_2mom', 'iwgan_4mom', 'iwgan_10mom',
            'mmdgan_2mom', 'mmdgan_4mom', 'mmdgan_10mom']

# Experiments corresponds to a specific model on a certain data dimensionality.
for experiment in log_dirs:
    means = []
    stds = []
    # Each base directory is a repeated instance of that model.
    for base_dir in base_dirs:
        scores_all = np.loadtxt(os.path.join(base_dir, experiment, 'scores.txt'))
        scores_not_nan = scores_all[~np.isnan(scores_all)]
        scores = scores_not_nan[-10:]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    # Print summary statistic for the multiple runs of each experiment.
    print('{}: {:.4f} +- {:.4f}'.format(experiment, np.mean(means), np.std(means)))
