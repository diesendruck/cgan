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
base_path = '/home/maurice/iwgn_multivariate/results'
model_run_names = [n for n in os.listdir(base_path) if model_name in n]

FIXED_DIM_CHOICES = [2, 4, 10]  # Defined by generative scripts.

print('Results: {}'.format(model_name))
for dim in FIXED_DIM_CHOICES:
    model_dim_combo = '{}_dim{}'.format(model_name, dim)
    # Fetch only runs for that model and dim.
    # e.g. checking if the identifier "ce_iw_dim2" is in "ce_iw_dim2_run5".
    runs = [name for name in model_run_names if model_dim_combo in name]
    mmd_run_means = []
    energy_run_means = []
    kl_run_means = []
    for run in runs:
        tail = 5
        # Fetch and store MMD results.
        mmd_run_scores_all = np.loadtxt(os.path.join(base_path, run, 'scores_mmd.txt'))
        mmd_run_scores_not_nan = mmd_run_scores_all[~np.isnan(mmd_run_scores_all)]
        mmd_run_scores = mmd_run_scores_not_nan[-10:]
        mmd_run_means.append(np.mean(mmd_run_scores))
        # Fetch and store energy results.
        energy_run_scores_all = np.loadtxt(os.path.join(base_path, run, 'scores_energy.txt'))
        energy_run_scores_not_nan = energy_run_scores_all[~np.isnan(energy_run_scores_all)]
        energy_run_scores = energy_run_scores_not_nan[-tail:]
        energy_run_means.append(np.mean(energy_run_scores))
        # Fetch and store KL results.
        kl_run_scores_all = np.loadtxt(os.path.join(base_path, run, 'scores_kl.txt'))
        kl_run_scores_not_nan = kl_run_scores_all[~np.isnan(kl_run_scores_all)]
        kl_run_scores = kl_run_scores_not_nan[-tail:]
        kl_run_means.append(np.mean(kl_run_scores))

    # Print summary statistic for the multiple runs of each experiment.
    print('  {} (n={}): MMD {:.4f} +- {:.4f},  E: {:.4f} +- {:.4f},  KL: {:.4f} +- {:.4f}'.format(
        model_dim_combo, len(mmd_run_means),
        np.mean(mmd_run_means), np.std(mmd_run_means),
        np.mean(energy_run_means), np.std(energy_run_means),
        np.mean(kl_run_means), np.std(kl_run_means)))
print
