import tensorflow as tf
layers = tf.layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pdb
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys


tag = 'bar'
data_set = 'bar'
data_num = 10000
mb_size = 1024  # 128
z_dim = 10  # 5
x_dim = 1
y_dim = 1
h_dim = 5
learning_rate = 1e-4
log_iter = 1000
log_dir = 'cgan_out_{}'.format(tag)
max_iter = 100000


def generate_data(n):
    def gen_from_filled_circle():
        radian = np.random.uniform(0., 2. * np.pi)
        max_radius = 1.5
        v1 = np.random.uniform(0., 2 * max_radius) * np.cos(radian)
        v2 = np.random.uniform(0., max_radius) * np.sin(radian)
        out = np.reshape([v1, v2], [1, -1])
        return out

    def gen_from_angled_bar():
        v1 = np.random.uniform(0., 6)
        #v2 = np.random.normal(0., 2. - v1 / 3.)
        v2 = np.random.normal(0.,  v1 / 6.)
        out = np.reshape([v1, v2], [1, -1])
        return out

    def gen_from_horseshoe():
        # Consider a latent variable that regulates height and income.
        latent = np.random.gamma(1., 8.)
        v1_mean = -0.00003 * np.exp(-1.0 * (0.13 * latent - 12)) + 5.5
        v1 = np.random.normal(v1_mean, 0.1)  # Height
        v2 = latent / 50 * 50000 + \
            abs(np.random.normal(0, 9999. / (1. + np.exp(0.15*(latent - 40.))) + 1.))  # Income
        out = np.reshape([v1, v2], [1, -1])
        return out

    if data_set == 'circle':
        sampling_fn = gen_from_filled_circle
    elif data_set == 'bar':
        sampling_fn = gen_from_angled_bar
    elif data_set == 'horseshoe':
        sampling_fn = gen_from_horseshoe
    else:
        raise ValueError('data_set selection not found')
        sys.exit()

    data_raw_unthinned = np.zeros((n, 2))
    data_raw = sampling_fn()
    for i in range(n):
        data_raw_unthinned[i] = sampling_fn()
    while len(data_raw) < n:
        out_xyl = sampling_fn()
        if np.random.binomial(1, thinning_fn(out_xyl[0][0], is_tf=False)):
            data_raw = np.concatenate((data_raw, out_xyl), axis=0)

    data_raw_mean = np.mean(data_raw, axis=0)
    data_raw_std = np.std(data_raw, axis=0)
    data_normed = (data_raw - data_raw_mean) / data_raw_std
    return data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned


def sample_data(data, batch_size):
    assert data.shape[1] == 2, 'data shape not 2'
    idxs = np.random.choice(data_num, batch_size)
    batch_x = np.reshape(data[idxs, 0], [-1, 1])
    batch_y = np.reshape(data[idxs, 1], [-1, 1])
    return batch_x, batch_y


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def compute_mmd(arr1, arr2, sigma_list=None, use_tf=False):
    """Computes mmd between two numpy arrays of same size."""
    if sigma_list is None:
        sigma_list = [1.0]

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


def plot(generated, data_raw, data_raw_unthinned, it, mmd_gen_vs_unthinned):
    gen_v1 = generated[:, 0]
    gen_v2 = generated[:, 1]
    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[1] for d in data_raw_unthinned]

    # Will use normalized data for evaluation of D.
    data_normed = (data_raw - data_raw_mean) / data_raw_std

    # Evaluate D on grid.
    grid_gran = 20
    grid_x = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)
    grid_y = np.linspace(min(data_raw[:, 1]), max(data_raw[:, 1]), grid_gran)
    vals_on_grid = np.zeros((grid_gran, grid_gran))
    for i in range(grid_gran):
        for j in range(grid_gran):
            grid_x_normed = (grid_x[i] - data_raw_mean[0]) / data_raw_std[0]
            grid_y_normed = (grid_y[j] - data_raw_mean[0]) / data_raw_std[0]
            vals_on_grid[i][j] = run_discrim(grid_x_normed, grid_y_normed)

    fig = plt.figure()
    gs = GridSpec(8, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.1)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    ax_joint.set_aspect('auto')
    ax_joint.imshow(vals_on_grid, interpolation='nearest', origin='lower', alpha=0.3, aspect='auto',
        extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    ax_thinning = ax_joint.twinx()
    ax_thinning.plot(grid_x, thinning_fn(grid_x, is_tf=False), color='red', alpha=0.3)
    ax_marg_x.hist([raw_v1, gen_v1], bins=30, color=['gray', 'blue'],
        label=['data', 'gen'], alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=30, color=['gray', 'blue'],
        label=['data', 'gen'], orientation="horizontal", alpha=0.3, normed=True)
    ax_marg_x.legend()
    ax_marg_y.legend()

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    #ax_joint.set_xlabel('Joint: height (ft)')
    #ax_joint.set_ylabel('Joint: income ($)')

    # Set labels on marginals
    #ax_marg_y.set_xlabel('Marginal: income')
    #ax_marg_x.set_ylabel('Marginal: height')

    ########
    # EVEN MORE PLOTTING.
    ax_raw = fig.add_subplot(gs[5:8, 0:3], sharex=ax_joint)
    ax_raw_marg_x = fig.add_subplot(gs[4, 0:3], sharex=ax_raw)
    ax_raw_marg_y = fig.add_subplot(gs[5:8, 3], sharey=ax_raw)
    ax_raw.scatter(raw_unthinned_v1, raw_unthinned_v2, c='gray', alpha=0.1)
    ax_raw_marg_x.hist(raw_unthinned_v1, bins=30, color='gray',
        label='d', alpha=0.3, normed=True)
    ax_raw_marg_y.hist(raw_unthinned_v2, bins=30, color='gray',
        label='d', orientation="horizontal", alpha=0.3, normed=True)
    plt.setp(ax_raw_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_raw_marg_y.get_yticklabels(), visible=False)
    ########

    plt.suptitle('cgan. it: {}, mmd_gen_vs_unthinned: {}'.format(it, mmd_gen_vs_unthinned))

    plt.savefig('{}/{}.png'.format(log_dir, it))
    plt.close()


################################################################################
# BEGIN: Build model.
def dense(x, width, activation, batch_residual=False):
    if not batch_residual:
        x_ = layers.dense(x, width, activation=activation)
        return layers.batch_normalization(x_)
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        return layers.batch_normalization(x_) + x


def discriminator(x, y, reuse=False):
    inputs = tf.concat(axis=1, values=[x, y])
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def generator(z, x, reuse=False):
    inputs = tf.concat(axis=1, values=[z, x])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, y_dim, activation=None)
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def run_discrim(x_in, y_in):
    x_in = np.reshape(x_in, [-1, 1])
    y_in = np.reshape(y_in, [-1, 1])
    return sess.run(d_real, feed_dict={x: x_in, y: y_in}) 


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def thinning_fn(inputs, is_tf=True): 
    """In 2D case, thinning based on x only. Inputs is a vector of x values."""
    if is_tf:
        return 0.99 / (1. + tf.exp(-0.95 * (inputs - 3.))) + 0.01
    else:
        return 0.99 / (1. + np.exp(-0.95 * (inputs - 3.))) + 0.01


# Beginning of graph.
z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
y = tf.placeholder(tf.float32, shape=[None, y_dim], name='y')

g, g_vars = generator(z, x, reuse=False)
d_real, d_logit_real, d_vars = discriminator(x, y, reuse=False)
d_fake, d_logit_fake, _ = discriminator(x, g, reuse=True)

errors_real = sigmoid_cross_entropy_with_logits(d_logit_real,
    tf.ones_like(d_logit_real))
errors_fake = sigmoid_cross_entropy_with_logits(d_logit_fake,
    tf.zeros_like(d_logit_fake))
d_loss_real = tf.reduce_mean(errors_real)
d_loss_fake = tf.reduce_mean(errors_fake)

# Assemble losses.
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

# Set optim nodes.
clip = 0
if clip:
    d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    d_grads_, d_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=d_vars))
    d_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in d_grads_])
    d_optim = d_opt.apply_gradients(zip(d_grads_clipped_, d_vars_))
else:
    d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    g_loss, var_list=g_vars)

# End: Build model.
################################################################################

# Load data.
data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned = generate_data(data_num)

# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
for it in range(max_iter):
    x_batch, y_batch = sample_data(data_normed, mb_size)
    z_batch = get_sample_z(mb_size, z_dim)

    for _ in range(5):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                [d_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            feed_dict={
                x: x_batch,
                z: z_batch,
                y: y_batch})
    for _ in range(1):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                [g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            feed_dict={
                x: x_batch,
                z: z_batch,
                y: y_batch})

    if it % log_iter == 0:
        n_sample = 10000
        z_sample = get_sample_z(n_sample, z_dim)

        # SAMPLE FROM CONDITIONAL, REGULATED BY THINNING_FN.
        raw_marg_x = data_raw[:, 0]
        thinning_fn_x = thinning_fn(raw_marg_x, is_tf=False)
        weights_x = 1. / thinning_fn_x
        weights_x_sum_normalized = weights_x / np.sum(weights_x)
        x_sample_unnormed = np.random.choice(raw_marg_x, size=n_sample,
            p=weights_x_sum_normalized)
        x_sample = np.reshape(
            (x_sample_unnormed - data_raw_mean[0]) / data_raw_std[0],  # Normed.
            [-1, 1])

        g_out = sess.run(g, feed_dict={z: z_sample[:len(x_sample)], x: x_sample})
        generated_normed = np.hstack((x_sample, g_out))
        generated = np.array(generated_normed) * data_raw_std[:2] + data_raw_mean[:2]

        mmd_gen_vs_unthinned, _ = compute_mmd(
            generated[np.random.choice(n_sample, 500)],
            data_raw_unthinned[np.random.choice(data_num, 500)])

        fig = plot(generated, data_raw, data_raw_unthinned, it, mmd_gen_vs_unthinned)

        # Print diagnostics.
        print("#################")
        print('Iter: {}'.format(it))
        print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))
        print('  mmd_gen_vs_unthinned: {:.4}'.format(mmd_gen_vs_unthinned))
        print(data_raw[np.random.choice(data_num, 5), :])
        print
        print(generated[:5])

