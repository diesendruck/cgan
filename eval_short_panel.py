import argparse
import numpy as np
import os
import pdb
import sys


"""Runs eval on specified model, and prints output to console.

  Takes model name as command line argument. e.g.
  python eval_short_panels.py "MODEL_NAME"

  Valid model names are defined in run_short_pane.sh, and are as follows:
    ['ce_iw', 'ce_sn', 'ce_miw',
     'mmd_iw', 'mmd_sn', 'mmd_miw',
     'cgan', 'upsample']

"""

model_name = sys.argv[1]
base_path = '/home/maurice/cgan/results'
model_trials = [d for d in os.listdir(base_path) if model_name in d]

FIXED_DIM_CHOICES = [2, 4, 10]  # Defined by generative scripts.

print('Results: {}'.format(model_name))
for dim in FIXED_DIM_CHOICES:
    model_with_dim = '{}_dim{}'.format(model_name, dim)
    # Fetch only trial runs for that model and dim.
    # e.g. checking if "ce_iw_dim2" in "ce_iw_dim2_run5"
    trials = [trial for trial in model_trials if model_with_dim in trial]
    mmd_means = []
    ksd_means = []
    for trial in trials:
        # Fetch and store MMD results.
        scores_all = np.loadtxt(os.path.join(base_path, trial, 'scores_mmd.txt'))
        scores_not_nan = scores_all[~np.isnan(scores_all)]
        scores = scores_not_nan[-10:]
        mmd_means.append(np.mean(scores))
        # Fetch and store KSD results.
        #scores_all = np.loadtxt(os.path.join(base_path, trial, 'scores_ksd.txt'))
        #scores_not_nan = scores_all[~np.isnan(scores_all)]
        #scores = scores_not_nan[-10:]
        #ksd_means.append(np.mean(scores))

    # Print summary statistic for the multiple runs of each experiment.
    print('  MMD: {}: {:.4f} +- {:.4f}'.format(
        model_with_dim, np.mean(mmd_means), np.std(mmd_means)))
    print('  KSD: {}: {:.4f} +- {:.4f}'.format(
        model_with_dim, np.mean(ksd_means), np.std(ksd_means)))
print
