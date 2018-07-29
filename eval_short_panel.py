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
all_trials = [d for d in os.listdir(base_path) if model_name in d]

FIXED_DIM_CHOICES = [2, 4, 10]  # Defined by generative scripts.

for dim in FIXED_DIM_CHOICES:
    model_with_dim = '{}_dim{}'.format(model_name, dim)
    # Fetch only trial runs for that model and dim.
    # e.g. checking if "ce_iw_dim2" in "ce_iw_dim2_run5"
    trials = [trial for trial in all_trials if model_with_dim in trial]
    means = []
    stds = []
    for trial in trials:
        scores_all = np.loadtxt(os.path.join(base_path, trial, 'scores.txt'))
        scores_not_nan = scores_all[~np.isnan(scores_all)]
        scores = scores_not_nan[-10:]
        means.append(np.mean(scores))
        stds.append(np.std(scores))
    # Print summary statistic for the multiple runs of each experiment.
    print('{}: {:.4f} +- {:.4f}'.format(
        model_with_dim, np.mean(means), np.std(means)))
print
