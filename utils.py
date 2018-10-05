import argparse
import tensorflow as tf
layers = tf.layers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb
import scipy.stats as stats

from matplotlib.gridspec import GridSpec
from tensorflow.examples.tutorials.mnist import input_data


def one_time_data_setup():

    if not os.path.exists('data'):
        os.makedirs('data')

    gen_type = 1
    def _make_files(n, data_dim, latent_dim, with_latents, m_weight):
        if gen_type == 1:
            (data_raw, data_raw_weights,
             data_raw_unthinned, data_raw_unthinned_weights,
             data_normed, data_raw_mean, data_raw_std) = \
                generate_data(
                     n, data_dim, latent_dim, with_latents=with_latents,
                     m_weight=m_weight)
        elif gen_type == 2:
            (data_raw, data_raw_weights,
             data_raw_unthinned, data_raw_unthinned_weights,
             data_normed, data_raw_mean, data_raw_std) = \
                generate_data2(n, data_dim, m_weight=m_weight)

        np.save('data/m_weight.npy', m_weight)
        np.save('data/{}d_data_raw.npy'.format(data_dim), data_raw[:, -data_dim:])  # No latents.
        np.save('data/{}d_data_raw_weights.npy'.format(data_dim), data_raw_weights)
        np.save('data/{}d_data_raw_unthinned.npy'.format(data_dim), data_raw_unthinned[:, -data_dim:])  # No latents.
        np.save('data/{}d_data_raw_unthinned_weights.npy'.format(data_dim), data_raw_unthinned_weights)
        np.save('data/{}d_data_normed.npy'.format(data_dim), data_normed[:, -data_dim:])  # No latents.
        np.save('data/{}d_data_raw_mean.npy'.format(data_dim), data_raw_mean[-data_dim:])  # No latents.
        np.save('data/{}d_data_raw_std.npy'.format(data_dim), data_raw_std[-data_dim:])  # No latents.

        np.save('data/with_latents_{}d_data_raw.npy'.format(data_dim), data_raw)
        np.save('data/with_latents_{}d_data_raw_weights.npy'.format(data_dim), data_raw_weights)
        np.save('data/with_latents_{}d_data_raw_unthinned.npy'.format(data_dim), data_raw_unthinned)
        np.save('data/with_latents_{}d_data_raw_unthinned_weights.npy'.format(data_dim), data_raw_unthinned_weights)
        np.save('data/with_latents_{}d_data_normed.npy'.format(data_dim), data_normed)
        np.save('data/with_latents_{}d_data_raw_mean.npy'.format(data_dim), data_raw_mean)
        np.save('data/with_latents_{}d_data_raw_std.npy'.format(data_dim), data_raw_std)

    n = 10000
    latent_dim = 100 
    with_latents = True
    # Fetch constant, which is based on thinning fn.
    m_weight = thinning_fn([0], return_normalizing_constant=True)

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
            # Compute weight, based on latent.
            rand_latent = np.random.uniform(0, 1, latent_dim)
            rand_transformed = np.dot(rand_latent, fixed_transform)
            latent_weight = 1. / thinning_fn(rand_latent, m_weight=m_weight)
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
            thinning_value = thinning_fn(rand_latent, m_weight=1.)  # Strictly T, not M.
            to_use = np.random.binomial(1, thinning_value)
            if to_use:
                # Point was included.
                rand_transformed = np.dot(rand_latent, fixed_transform)

                latent_weight = 1. / thinning_fn(rand_latent, m_weight=m_weight)

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
        weights = vert(stats.beta.pdf(latent[:, 0], alpha, 1.))
        weights_unthinned = vert(stats.beta.pdf(latent_unthinned[:, 0], alpha, 1.))

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


def generate_data2(data_num, data_dim, m_weight=1):
    def _gen_2d(n):
        print('Making {}d data with truncated normal + thinning_fn.'.format(data_dim))
        ##################################################################
        # Sample unthinned data, as bimodal bivariate Gaussian.
        tnorm1 = stats.truncnorm(-5, 5, loc=-2, scale=1)
        tnorm2 = stats.truncnorm(-5, 5, loc=2, scale=1)
        data_raw_unthinned = np.concatenate(
            (tnorm1.rvs((n/2, data_dim)),
             tnorm2.rvs((n/2, data_dim))),
            axis=0)
        data_raw_unthinned_weights = \
            [1. / thinning_fn(d, m_weight=m_weight) for d in data_raw_unthinned] 

        ##################################################################
        # Sample raw data (thinned), starting with truncated normal draw, and
        # accepting with probability according to thinning function.
        data_raw = np.zeros((n, data_dim))
        data_raw_weights = np.zeros((n, 1))
        count = 0
        while count < n:
            #if count % 1000 == 0:  # Used to track growth of large data sets.
            #    print('Count: {}'.format(count))
            # Randomly select a point from one of the clusters.
            if np.random.uniform() < 0.5:
                rand_point = tnorm1.rvs((1, data_dim))[0]
            else:
                rand_point = tnorm2.rvs((1, data_dim))[0]
            # Get thinning value. Recall, thinning_fn is M, and we want T -- so
            # use m_weight = 1. Also, thinning_fn() thins on the 0th index of
            # the input, which is the first of the two data dimensions.
            thinning_value = thinning_fn(rand_point, m_weight=1.)
            to_use = np.random.binomial(1, thinning_value)
            if to_use:
                # Add the point to the collection.
                latent_weight = 1. / thinning_fn(rand_point, m_weight=m_weight)
                data_raw[count] = rand_point
                data_raw_weights[count] = latent_weight
                count += 1

        return data_raw, data_raw_weights, data_raw_unthinned, data_raw_unthinned_weights

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


def thinning_fn(inputs, m_weight=1, return_normalizing_constant=False):
    """Thinning on zero'th index of input."""
    eps = 1e-10

    case = 0

    if case == 0:
        # Used with generate_data().
        # Example: For thinning fn x^4 to integrate to 1 on [0,1], m_weight = 1./5.
        normalizing_constant = 1. / 5.
        if return_normalizing_constant:
            return normalizing_constant

        return (1. / m_weight) * inputs[0] ** 4 + eps
    elif case == 1:
        # Used with generate_data().
        # Example: For thinning fn x^8 to integrate to 1 on [0,1], m_weight = 1./9.
        normalizing_constant = 1. / 9.
        if return_normalizing_constant:
            return normalizing_constant

        return (1. / m_weight) * inputs[0] ** 8 + eps
    elif case == 2:
        # Used with generate_data().
        # Example: Consider thinning fn
        #   {y=0.01 for x on [0,0.5], y=1 for x on (0.5, 1]},
        # which integrates to 0.505 on [0,1]. To integrate to 1, m_weight = 0.505
        normalizing_constant = 0.505
        if return_normalizing_constant:
            return normalizing_constant

        if inputs[0] <= 0.5:
            return (1. / m_weight) * 0.01 * inputs[0]
        elif inputs[0] > 0.5:
            return (1. / m_weight) * 1. * inputs[0]
    #elif case == 3:
    #    # Example: For thinning fn ((x+3)/6)^8 to integrate to 1 on [-3,3],
    #    # m_weight = 6. / 9.
    #    normalizing_constant = 6. / 9.
    #    if return_normalizing_constant:
    #        return normalizing_constant

    #    return (1. / m_weight) * ((inputs[0] + 3.) / 6.) ** 8 + eps  
    elif case == 4:
        # Used with generate_data2().
        # Example: For thinning fn (0.95 / (1 + exp(-x)) + 0.05 to integrate to
        # 1 on [-5,5], m_weight = 5.25.
        normalizing_constant = 5.25
        if return_normalizing_constant:
            return normalizing_constant

        return (1. / m_weight) * (0.95 / (1. + np.exp(-1.*inputs[0])) + 0.05) + eps  
    elif case == 5:
        # Used with generate_data2().
        # Example: For thinning fn (0.99 / (1 + exp(-2x)) + 0.01 to integrate to
        # 1 on [-5,5], m_weight = 5.05.
        normalizing_constant = 5.05
        if return_normalizing_constant:
            return normalizing_constant

        return (1. / m_weight) * (0.99 / (1. + np.exp(-2.*inputs[0])) + 0.01) + eps  


def vert(arr):
    return np.reshape(arr, [-1, 1])


def get_data(data_dim, with_latents=False):
    if with_latents:
        prepended = 'with_latents_{}d'.format(data_dim)
    else:
        prepended = '{}d'.format(data_dim)
    m_weight = np.load('data/m_weight.npy')
    data_raw = np.load('data/{}_data_raw.npy'.format(prepended))
    data_raw_weights = np.load('data/{}_data_raw_weights.npy'.format(prepended))
    data_raw_unthinned = np.load('data/{}_data_raw_unthinned.npy'.format(prepended))
    data_raw_unthinned_weights = np.load('data/{}_data_raw_unthinned_weights.npy'.format(prepended))
    data_normed = np.load('data/{}_data_normed.npy'.format(prepended))
    data_raw_mean = np.load('data/{}_data_raw_mean.npy'.format(prepended))
    data_raw_std = np.load('data/{}_data_raw_std.npy'.format(prepended))
    return (m_weight, data_raw, data_raw_weights,
            data_raw_unthinned, data_raw_unthinned_weights,
            data_normed, data_raw_mean, data_raw_std)


def sample_data(data, data_weights, batch_size):
    idxs = np.random.choice(len(data), batch_size)
    batch_data = data[idxs]
    batch_weights = data_weights[idxs]
    return batch_data, batch_weights


def natural_sort(l): 
    import re
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def dense(x, width, activation, batch_residual=False, dropout=None):
    if batch_residual:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        x_ = layers.batch_normalization(x_) + x
    else:
        x_ = layers.dense(x, width, activation=activation)
        x_ = layers.batch_normalization(x_)

    if dropout:
        x_ = tf.nn.dropout(x_, dropout)
    return x_
    #if not batch_residual: 
    #    x_ = layers.dense(x, width, activation=activation) 
    #    r = layers.batch_normalization(x_) 
    #    return r 
    #else: 
    #    x_ = layers.dense(x, width, activation=activation, use_bias=False)
    #    r = layers.batch_normalization(x_) + x
    #    return r


def split_80_20(arr):
    n = len(arr)
    n80 = int(0.8 * n)
    n20 = n - n80
    arr80 = arr[:n80]
    arr20 = arr[n80:]
    return arr80, arr20


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


def plot_2d(generated, data_raw, data_raw_unthinned, step, measure_to_plot,
        log_dir):
    gen_v1 = generated[:, 0]
    gen_v2 = generated[:, 1]
    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[1] for d in data_raw_unthinned]

    ## Evaluate D on grid.
    #grid_gran = 20
    #grid_x = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)
    #grid_y = np.linspace(min(data_raw[:, 1]), max(data_raw[:, 1]), grid_gran)
    #vals_on_grid = np.zeros((grid_gran, grid_gran))
    #for i in range(grid_gran):
    #    for j in range(grid_gran):
    #        grid_x_normed = (grid_x[i] - data_raw_mean[0]) / data_raw_std[0]
    #        grid_y_normed = (grid_y[j] - data_raw_mean[0]) / data_raw_std[0]
    #        vals_on_grid[i][j] = run_discrim([grid_x_normed, grid_y_normed])

    fig = plt.figure()
    gs = GridSpec(8, 4)
    
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.1)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    
    ax_joint.set_aspect('auto')
    ax_joint.imshow(vals_on_grid, interpolation='nearest', origin='lower',
        alpha=0.3, aspect='auto',
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])

    bins = np.arange(-3, 3, 0.2)
    #ax_marg_x.hist([raw_v1, gen_v1], bins=bins, color=['gray', 'blue'],
    #    label=['data', 'gen'], alpha=0.3, normed=True)
    ax_marg_x.hist(raw_v1, bins=bins, color='gray',
        label='data', alpha=0.3, normed=True)
    ax_marg_x.hist(gen_v1, bins=bins, color='blue',
        label='gen', alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=bins, color=['gray', 'blue'],
        label=['data', 'gen'], orientation="horizontal", alpha=0.3, normed=True)
    ax_marg_x.legend()
    ax_marg_y.legend()

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    ########
    # EVEN MORE PLOTTING.
    ax_raw = fig.add_subplot(gs[5:8, 0:3], sharex=ax_joint)
    ax_raw_marg_x = fig.add_subplot(gs[4, 0:3], sharex=ax_raw)
    ax_raw_marg_y = fig.
    ax_raw.scatter(raw_unthinned_v1, raw_unthinned_v2, c='green', alpha=0.1)
    ax_raw_marg_x.hist(raw_unthinned_v1, bins=bins, color='green',
        label='d', alpha=0.3, normed=True)
    ax_raw_marg_y.hist(raw_unthinned_v2, bins=bins, color='green',
        label='d', orientation="horizontal", alpha=0.3, normed=True)
    plt.setp(ax_raw_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_raw_marg_y.get_yticklabels(), visible=False)
    ########

    plt.suptitle('iwgan. step: {}, discrepancy: {:.4f}'.format(
        step, measure_to_plot))
    
    plt.savefig('{}/{}.png'.format(log_dir, step))
    plt.close()


"""

# NOTE: The following must be commented out after data generation, before 
#   training any models.
#one_time_data_setup()
