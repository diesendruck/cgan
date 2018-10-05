import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
sys.path.append('/home/maurice/mmd')
import scipy.stats as stats
import tensorflow as tf
layers = tf.layers
import time

from matplotlib.gridspec import GridSpec
from tensorflow.examples.tutorials.mnist import input_data

from kl_estimators import naive_estimator as compute_kl
from mmd_utils import compute_mmd, compute_energy
from utils import get_data, generate_data, sample_data, dense, split_80_20


for data_dim in [2, 4, 10]:
    (m_weight,
     data_raw,
     data_raw_weights,
     data_raw_unthinned,
     data_raw_unthinned_weights,
     data_normed,
     data_raw_mean,
     data_raw_std) = get_data(data_dim, with_latents=False)

    # To do model selection, separate out two sets of unthinned data. Use the
    # first to select the model, and the second to report that model's
    # performance. Since data_raw_unthinned is sampled entirely separately from
    # training data, and is not used in training, it will be used for validation
    # and test sets.
    (data_raw_unthinned_validation,
     data_raw_unthinned_test) = split_80_20(data_raw_unthinned)


    def compute_discrepancies(cand_set, ref_set):
        """Computes discrepancies between two sets."""
        cand_set_n = len(cand_set)
        ref_set_n = len(ref_set)
        n_sample = 1000 
        assert ((cand_set_n >= n_sample) & (ref_set_n >= n_sample)), \
            'n_sample too high for one of the inputs.'
        # Compute MMD, Energy, and KL, between simulations and unthinned data.
        mmd_, _ = compute_mmd(
            cand_set[np.random.choice(cand_set_n, n_sample)],
            ref_set[np.random.choice(ref_set_n, n_sample)])
        energy_ = compute_energy(
            cand_set[np.random.choice(cand_set_n, n_sample)],
            ref_set[np.random.choice(ref_set_n, n_sample)])
        kl_ = compute_kl(
            cand_set[np.random.choice(cand_set_n, n_sample)],
            ref_set[np.random.choice(ref_set_n, n_sample)],
            k=5)
        return mmd_, energy_, kl_

    num_runs = 100
    r = np.zeros((num_runs, 3))
    for i in range(num_runs):
        (mmd_raw_vs_unthinned_v,
         energy_raw_vs_unthinned_v,
         kl_raw_vs_unthinned_v) = \
             compute_discrepancies(
                 data_raw, data_raw_unthinned_validation)
        r[i] = [mmd_raw_vs_unthinned_v,
                energy_raw_vs_unthinned_v,
                kl_raw_vs_unthinned_v]

    mmd_, energy_, kl_ = np.mean(r, axis=0)

    print((
        'baseline dim{}:\n'
        '{:.5f} mmd2\n'
        '{:.5f} energy\n'
        '{:.5f} kl\n').format(data_dim, mmd_, energy_, kl_))

pdb.set_trace()
