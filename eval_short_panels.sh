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

model_to_eval = sys.argv[1]
base_path = '/home/maurice/cgan/results'
log_dirs = [d for d in os.listdirs(base_path) if model_to_eval in d
pdb.set_trace()
# All models together.

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
