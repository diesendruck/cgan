import argparse
import tensorflow as tf
layers = tf.layers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb

from scipy.stats import beta
from tensorflow.examples.tutorials.mnist import input_data


def one_time_data_setup():

    if not os.path.exists('data'):
        os.makedirs('data')

    def _make_files(n, data_dim, latent_dim, with_latents, m_weight):
        (data_raw, data_raw_weights,
         data_raw_unthinned, data_raw_unthinned_weights,
         data_normed, data_raw_mean, data_raw_std) = generate_data(
                 n, data_dim, latent_dim, with_latents=with_latents, m_weight=m_weight)
        np.save('data/{}d_data_raw.npy'.format(data_dim), data_raw[:, -data_dim:])  # Removing latents.
        np.save('data/{}d_data_raw_weights.npy'.format(data_dim), data_raw_weights)
        np.save('data/{}d_data_raw_unthinned.npy'.format(data_dim), data_raw_unthinned[:, -data_dim:])  # Removing latents.
        np.save('data/{}d_data_raw_unthinned_weights.npy'.format(data_dim), data_raw_unthinned_weights)
        np.save('data/{}d_data_normed.npy'.format(data_dim), data_normed[:, -data_dim:])  # Removing latents.
        np.save('data/{}d_data_raw_mean.npy'.format(data_dim), data_raw_mean[-data_dim:])  # Removing latents.
        np.save('data/{}d_data_raw_std.npy'.format(data_dim), data_raw_std[-data_dim:])  # Removing latents.

        np.save('data/with_latents_{}d_data_raw.npy'.format(data_dim), data_raw)
        np.save('data/with_latents_{}d_data_raw_weights.npy'.format(data_dim), data_raw_weights)
        np.save('data/with_latents_{}d_data_raw_unthinned.npy'.format(data_dim), data_raw_unthinned)
        np.save('data/with_latents_{}d_data_raw_unthinned_weights.npy'.format(data_dim), data_raw_unthinned_weights)
        np.save('data/with_latents_{}d_data_normed.npy'.format(data_dim), data_normed)
        np.save('data/with_latents_{}d_data_raw_mean.npy'.format(data_dim), data_raw_mean)
        np.save('data/with_latents_{}d_data_raw_std.npy'.format(data_dim), data_raw_std)

    n = 10000
    latent_dim = 10
    with_latents = True
    m_weight = 5.  # NOTE: THIS MUST AGREE WITH THE THINNING_FN DEFINITION!

    # Make the files and store them in the "data" subdirectory.
    _make_files(n, 2, latent_dim, with_latents=True, m_weight=m_weight)
    _make_files(n, 4, latent_dim, with_latents=True, m_weight=m_weight)
    _make_files(n, 10, latent_dim, with_latents=True, m_weight=m_weight)

    print('Data generation complete. COMMENT OUT "one_time_data_setup()" in '
          'utils.py before proceeding to training.')


def generate_data(data_num, data_dim, latent_dim, with_latents=False, m_weight=1):
    def _gen_2d(n):
        print('Making {}d data with uniform/thinning_fn.'.format(data_dim))
        ##################################################################
        # Sample unthinned data, as Uniform latents with Normal transform.
        data_raw_unthinned = np.zeros((n, data_dim))
        data_raw_unthinned_latents = np.zeros((n, latent_dim))
        data_raw_unthinned_weights = np.zeros((n, 1))
        fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))
        for i in range(n):
            # Sample a latent uniform variable.
            # Apply the Normal transform.
            # Compute weight, based on 0th index of latent.
            rand_latent = np.random.uniform(0, 1, latent_dim)
            rand_transformed = np.dot(rand_latent, fixed_transform)
            latent_weight = 1. / thinning_fn(rand_latent, is_tf=False, m_weight=m_weight)
            # Store results.
            data_raw_unthinned[i] = rand_transformed
            data_raw_unthinned_latents[i] = rand_latent
            data_raw_unthinned_weights[i] = latent_weight

        ##################################################################
        # Sample raw data (thinned), starting with Uniform draw, and accepting
        #   with probability according to thinning function.
        data_raw = np.zeros((n, data_dim))
        data_raw_latents = np.zeros((n, latent_dim))
        data_raw_weights = np.zeros((n, 1))
        count = 0
        while count < n:
            rand_latent = np.random.uniform(0, 1, latent_dim)
            thinning_value = thinning_fn(rand_latent, is_tf=False, m_weight=1.)  # Strictly T, not M.
            to_use = np.random.binomial(1, thinning_value)
            if to_use:
                # Point was included.
                rand_transformed = np.dot(rand_latent, fixed_transform)

                # TODO: Should this be 1 / t(x)?
                #latent_weight = thinning_fn(rand_latent, is_tf=False, m_weight=m_weight)
                latent_weight = 1. / thinning_fn(rand_latent, is_tf=False, m_weight=m_weight)

                # Add the point to the collection.
                data_raw[count] = rand_transformed
                data_raw_latents[count] = rand_latent
                data_raw_weights[count] = latent_weight
                count += 1

        if with_latents:
            data_raw = np.concatenate((data_raw_latents, data_raw), axis=1)
            data_raw_unthinned = np.concatenate(
                (data_raw_unthinned_latents, data_raw_unthinned), axis=1)

        return data_raw, data_raw_weights, data_raw_unthinned, data_raw_unthinned_weights

    def _gen_beta_2d(n):
        print('Making {}d data with beta/1-over-beta.'.format(data_dim))
        alpha = 0.001
        beta_params = [1] * latent_dim
        beta_params[0] = alpha

        latent = np.random.uniform(0, 1, size=(data_num, latent_dim))
        latent_unthinned = np.random.beta(beta_params, beta_params, (data_num, latent_dim))
        weights = vert(beta.pdf(latent[:, 0], alpha, 1.))
        weights_unthinned = vert(beta.pdf(latent_unthinned[:, 0], alpha, 1.))

        fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))
        data = np.dot(latent, fixed_transform)
        data_unthinned = np.dot(latent_unthinned, fixed_transform)

        if with_latents:
            data = np.concatenate((latent, data), axis=1)
            data_unthinned = np.concatenate(
                (latent_unthinned, data_unthinned), axis=1)

        return data, weights, data_unthinned, weights_unthinned 

    beta_thinning = False
    if beta_thinning:
        (data_raw,
         data_raw_weights,
         data_raw_unthinned,
         data_raw_unthinned_weights) = _gen_beta_2d(data_num)
    else:
        (data_raw,
         data_raw_weights,
         data_raw_unthinned,
         data_raw_unthinned_weights) = _gen_2d(data_num)

    data_raw_mean = np.mean(data_raw, axis=0)
    data_raw_std = np.std(data_raw, axis=0)
    data_normed = (data_raw - data_raw_mean) / data_raw_std

    return (data_raw, data_raw_weights,
            data_raw_unthinned, data_raw_unthinned_weights,
            data_normed, data_raw_mean, data_raw_std)


def thinning_fn(inputs, is_tf=True, m_weight=1):
    """Thinning on zero'th index of input."""
    eps = 1e-10
    if is_tf:
        # NOTE: THIS MUST AGREE WITH THE m_weight!
        #   e.g. For this to integrate to 1 on [0,1], m_weight = 5.
        return m_weight * inputs[0] ** 4 + eps  
    else:
        return m_weight * inputs[0] ** 4 + eps


def vert(arr):
    return np.reshape(arr, [-1, 1])


def get_data(data_dim, with_latents=False):
    if with_latents:
        prepended = 'with_latents_{}d'.format(data_dim)
    else:
        prepended = '{}d'.format(data_dim)
    data_raw = np.load('data/{}_data_raw.npy'.format(prepended))
    data_raw_weights = np.load('data/{}_data_raw_weights.npy'.format(prepended))
    data_raw_unthinned = np.load('data/{}_data_raw_unthinned.npy'.format(prepended))
    data_raw_unthinned_weights = np.load('data/{}_data_raw_unthinned_weights.npy'.format(prepended))
    data_normed = np.load('data/{}_data_normed.npy'.format(prepended))
    data_raw_mean = np.load('data/{}_data_raw_mean.npy'.format(prepended))
    data_raw_std = np.load('data/{}_data_raw_std.npy'.format(prepended))
    return (data_raw, data_raw_weights,
            data_raw_unthinned, data_raw_unthinned_weights,
            data_normed, data_raw_mean, data_raw_std)


def sample_data(data, data_weights, batch_size):
    idxs = np.random.choice(len(data), batch_size)
    batch_data = data[idxs]
    batch_weights = data_weights[idxs]
    return batch_data, batch_weights


"""
def compute_mmd(arr1, arr2, sigma_list=None, use_tf=False):
    #Computes mmd between two numpy arrays of same size.
    if sigma_list is None:
        sigma_list = [0.1, 1.0, 10.0]

    n1 = len(arr1)
    n2 = len(arr2)

    if use_tf:
        v = tf.concat([arr1, arr2], 0)
        VVT = tf.matmul(v, tf.transpose(v))
        sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
        sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
        exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += tf.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = tf.matrix_band_part(K_xx, 0, -1)
        K_yy_upper = tf.matrix_band_part(K_yy, 0, -1)
        num_combos_x = tf.to_float(n1 * (n1 - 1) / 2)
        num_combos_y = tf.to_float(n2 * (n2 - 1) / 2)
        num_combos_xy = tf.to_float(n1 * n2)
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_x +
               tf.reduce_sum(K_yy_upper) / num_combos_y -
               2 * tf.reduce_sum(K_xy) / num_combos_xy)
        return mmd, exp_object
    else:
        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, [-1, 1])
            arr2 = np.reshape(arr2, [-1, 1])
        v = np.concatenate((arr1, arr2), 0)
        VVT = np.matmul(v, np.transpose(v))
        sqs = np.reshape(np.diag(VVT), [-1, 1])
        sqs_tiled_horiz = np.tile(sqs, np.transpose(sqs).shape)
        exp_object = sqs_tiled_horiz - 2 * VVT + np.transpose(sqs_tiled_horiz)
        K = 0.0
        for sigma in sigma_list:
            gamma = 1.0 / (2.0 * sigma**2)
            K += np.exp(-gamma * exp_object)
        K_xx = K[:n1, :n1]
        K_yy = K[n1:, n1:]
        K_xy = K[:n1, n1:]
        K_xx_upper = np.triu(K_xx)
        K_yy_upper = np.triu(K_yy)
        num_combos_x = n1 * (n1 - 1) / 2
        num_combos_y = n2 * (n2 - 1) / 2
        mmd = (np.sum(K_xx_upper) / num_combos_x +
               np.sum(K_yy_upper) / num_combos_y -
               2 * np.sum(K_xy) / (n1 * n2))
        return mmd, exp_object
"""

# NOTE: The following must be commented out after data generation, before 
#   training any models.
#one_time_data_setup()
