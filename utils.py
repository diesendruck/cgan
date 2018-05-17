import argparse
import tensorflow as tf
layers = tf.layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb


def generate_data(n, data_dim, latent_dim, with_latents=False, m_weight=1):
    def gen_2d(n):
        #latent_mean = np.zeros(latent_dim)
        #latent_cov = np.identity(latent_dim)
        fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))

        data_raw_unthinned = np.zeros((n, data_dim))
        data_raw_unthinned_latents = np.zeros((n, latent_dim))
        data_raw_unthinned_weights = np.zeros((n, 1))
        for i in range(n):
            #rand_latent = np.random.multivariate_normal(latent_mean, latent_cov)
            rand_latent = np.random.uniform(0, 1, latent_dim)
            rand_transformed = np.dot(rand_latent, fixed_transform)
            data_raw_unthinned[i] = rand_transformed
            data_raw_unthinned_latents[i] = rand_latent

            latent_weight = 1. / thinning_fn(rand_latent, is_tf=False, m_weight=m_weight)
            data_raw_unthinned_weights[i] = latent_weight

        data_raw = np.zeros((n, data_dim))
        data_raw_latents = np.zeros((n, latent_dim))
        data_raw_weights = np.zeros((n, 1))
        count = 0
        while count < n:
            #rand_latent = np.random.multivariate_normal(latent_mean, latent_cov)
            rand_latent = np.random.uniform(0, 1, latent_dim)
            thinning_value = thinning_fn(rand_latent, is_tf=False, m_weight=1.)  # Strictly T, not M.
            to_use = np.random.binomial(1, thinning_value)
            if to_use:
                rand_transformed = np.dot(rand_latent, fixed_transform)
                data_raw[count] = rand_transformed
                data_raw_latents[count] = rand_latent

                latent_weight = 1. / (m_weight * thinning_value)  # But weight needs m_weight.
                data_raw_weights[count] = latent_weight
                count += 1

        if with_latents:
            data_raw = np.concatenate((data_raw_latents, data_raw), axis=1)
            data_raw_unthinned = np.concatenate(
                (data_raw_unthinned_latents, data_raw_unthinned), axis=1)

        return data_raw, data_raw_weights, data_raw_unthinned, data_raw_unthinned_weights

    (data_raw,
     data_raw_weights,
     data_raw_unthinned,
     data_raw_unthinned_weights) = gen_2d(n)

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
        return m_weight * inputs[0] + eps
    else:
        return m_weight * inputs[0] + eps


def sample_data(data, data_weights, batch_size):
    idxs = np.random.choice(len(data), batch_size)
    batch_data = data[idxs]
    batch_weights = data_weights[idxs]
    return batch_data, batch_weights


def compute_mmd(arr1, arr2, sigma_list=None, use_tf=False):
    """Computes mmd between two numpy arrays of same size."""
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



