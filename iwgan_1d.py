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
import seaborn as sns
from matplotlib.gridspec import GridSpec


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--weighted', default=False, action='store_true', dest='weighted',
                    help='Chooses whether Vanilla GAN or IW-GAN.')
parser.add_argument('--do_p', default=False, action='store_true', dest='do_p',
                    help='Choose whether to use P, instead of TP')
parser.add_argument('--do_mmd', default=False, action='store_true', dest='do_mmd',
                    help='Choose whether to use MMD, instead of GAN')
args = parser.parse_args()
tag = args.tag
weighted = args.weighted
do_p = args.do_p
do_mmd = args.do_mmd
data_num = 10000
batch_size = 64
z_dim = 5
x_dim = 1
h_dim = 3
learning_rate = 1e-4
log_iter = 1000
log_dir = 'iwgan1d_out_{}'.format(tag)


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

    def gen_from_bimodal_gaussian():
        cluster_choice = np.random.binomial(n=1, p=0.5)
        if cluster_choice == 0:
            return np.random.normal(0, 0.5, size=(1, 1))
        else:
            return np.random.normal(2, 0.5, size=(1, 1))

    sampling_fn = gen_from_bimodal_gaussian

    data_raw_unthinned = np.zeros((n, 1))
    data_raw = sampling_fn()
    for i in range(n):
        data_raw_unthinned[i] = sampling_fn()
    while len(data_raw) < n:
        out = sampling_fn()
        if np.random.binomial(1, thinning_fn(out[0][0], is_tf=False)):
            data_raw = np.concatenate((data_raw, out), axis=0)

    data_raw_mean = np.mean(data_raw, axis=0)
    data_raw_std = np.std(data_raw, axis=0)
    data_normed = (data_raw - data_raw_mean) / data_raw_std 
    return data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned


def thinning_fn(inputs, is_tf=True):
    """Thinning on x only (height). Inputs is a vector of x values."""
    #if is_tf:
    #    return 0.99 / (1. + tf.exp(-0.95 * (inputs - 3.))) + 0.01
    #else:
    #    return 0.99 / (1. + np.exp(-0.95 * (inputs - 3.))) + 0.01
    if is_tf:
        return 0.5 / (1. + tf.exp(10 * (inputs - 1.))) + 0.5
    else:
        return 0.5 / (1. + np.exp(10 * (inputs - 1.))) + 0.5


# Load data.
data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned = \
    generate_data(data_num)
if do_p:
    data_normed = to_normed(data_raw_unthinned)
    data_raw = data_raw_unthinned


def sample_data(data, batch_size):
    assert data.shape[1] == 1, 'data shape not 1'
    idxs = np.random.choice(data_num, batch_size)
    batch_x = np.reshape(data[idxs, 0], [-1, 1])
    return batch_x


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def plot(generated, data_raw, data_raw_unthinned, it):
    gen_v1 = generated[:, 0] 
    raw_v1 = [d[0] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]

    # Will use normalized data for evaluation of D.
    data_normed = to_normed(data_raw)

    # Evaluate D on grid.
    grid_gran = 40
    grid_x = np.linspace(min(min(data_raw), min(generated)), max(max(data_raw), max(generated)), grid_gran)
    if not do_mmd:
        vals_on_grid = np.zeros(grid_gran)
        for i in range(grid_gran):
            grid_x_normed = (grid_x[i] - data_raw_mean) / data_raw_std
            vals_on_grid[i] = run_discrim(grid_x_normed)

    fig = plt.figure()
    gs = GridSpec(2, 1)
    ax_thinned = fig.add_subplot(gs[0, 0])
    ax_thinned.hist(raw_v1, color='gray', bins=30, label='d', alpha=0.3, normed=True)
    ax_thinned.hist(gen_v1, color='blue', bins=30, label='g', alpha=0.3, normed=True)
    ax_thinned.legend(loc='upper right')

    ax_unthinned = fig.add_subplot(gs[1, 0], sharex=ax_thinned)
    ax_unthinned.hist(raw_unthinned_v1, color='gray', bins=30, label='d', alpha=0.3, normed=True)
    ax_unthinned.legend(loc='upper right')

    ax_thinning_fn = ax_unthinned.twinx()
    ax_thinning_fn.plot(grid_x, thinning_fn(grid_x, is_tf=False), color='red', alpha=0.3, label='thinning')
    ax_thinning_fn.legend(loc='lower right')

    if not do_mmd:
        ax_discrim_fn = ax_thinned.twinx()
        ax_discrim_fn.plot(grid_x, vals_on_grid, color='green', alpha=0.3, label='discrim')
        ax_discrim_fn.legend(loc='lower right')


    # Turn off tick labels on marginals
    #plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    #plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    #ax_joint.set_xlabel('Joint: height (ft)')
    #ax_joint.set_ylabel('Joint: income ($)')

    # Set labels on marginals
    #ax_marg_y.set_xlabel('Marginal: income')
    #ax_marg_x.set_ylabel('Marginal: height')

    plt.savefig('{}/{}.png'.format(log_dir, it))
    plt.close()


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def run_discrim(x_in):
    x_in = np.reshape(x_in, [-1, 1])
    return sess.run(d_disc, feed_dict={x_disc: x_in}) 


def sample_generator(n_sample):
    z_sample_input = get_sample_z(n_sample, z_dim)
    g_out = sess.run(g_sample, feed_dict={z_sample: z_sample_input})
    generated = g_out * data_raw_std + data_raw_mean
    return generated


def to_raw(d, index=None):
    if index:
        return d * data_raw_std[index] + data_raw_mean[index]
    else:
        return d * data_raw_std + data_raw_mean


def to_normed(d, index=None):
    if index:
        return (d - data_raw_mean[index]) /  data_raw_std[index]
    else:
        return (d - data_raw_mean) /  data_raw_std


################################################################################
# BEGIN: Build model.
def dense(x, width, activation, batch_residual=False):
    if not batch_residual:
        x_ = layers.dense(x, width, activation=activation)
        return layers.batch_normalization(x_)
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        return layers.batch_normalization(x_) + x


def generator(z, reuse=False):
    #inputs = tf.concat(axis=1, values=[z, x])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(z, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, x_dim, activation=None)  # Outputing x.
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def compute_mmd(input1, input2, batch_size, weighted=False):
    """Computes MMD between two batches of d-dimensional inputs.
    
    In this setting, input1 is real and input2 is generated.
    """
    num_combos_xx = tf.to_float(batch_size * (batch_size - 1) / 2)
    num_combos_yy = tf.to_float(batch_size * (batch_size - 1) / 2)

    v = tf.concat([input1, input2], 0)
    VVT = tf.matmul(v, tf.transpose(v))
    sqs = tf.reshape(tf.diag_part(VVT), [-1, 1])
    sqs_tiled_horiz = tf.tile(sqs, tf.transpose(sqs).get_shape())
    exp_object = sqs_tiled_horiz - 2 * VVT + tf.transpose(sqs_tiled_horiz)

    K = 0
    sigma_list = [0.01, 1.0, 2.0]
    for sigma in sigma_list:
        #gamma = 1.0 / (2.0 * sigma ** 2)
        K += tf.exp(-0.5 * (1 / sigma) * exp_object)
    K_xx = K[:batch_size, :batch_size]
    K_yy = K[batch_size:, batch_size:]
    K_xy = K[:batch_size, batch_size:]
    K_xx_upper = tf.matrix_band_part(K_xx, 0, -1) - tf.matrix_band_part(K_xx, 0, 0)
    K_yy_upper = tf.matrix_band_part(K_yy, 0, -1) - tf.matrix_band_part(K_yy, 0, 0)
    K_yy_offdiag = K_yy - tf.matrix_band_part(K_yy, 0, 0)

    thinning_by_x = tf.reshape(thinning_fn(v[:, 0]), [-1, 1])
    thinning_by_x_tiled_horiz = tf.tile(thinning_by_x, [1, batch_size + batch_size])
    p1_weights = 1. / thinning_by_x_tiled_horiz
    p2_weights = tf.transpose(p1_weights)
    p1p2_weights_xx = p1_weights[:batch_size, :batch_size] * \
                      p2_weights[:batch_size, :batch_size]

    # Offdiag weighting.
    p1p2_weights_xx_offdiag = p1p2_weights_xx - \
                              tf.matrix_band_part(p1p2_weights_xx, 0, 0)
    p1p2_weights_xx_offdiag_normed = p1p2_weights_xx_offdiag / \
                                     tf.reduce_sum(p1p2_weights_xx_offdiag)
    Kw_xx_offdiag = K_xx * p1p2_weights_xx_offdiag_normed

    # Old weighting.
    p1p2_weights_xx_normed = p1p2_weights_xx / tf.reduce_sum(p1p2_weights_xx)
    Kw_xx = K_xx * p1p2_weights_xx_normed
    Kw_xx_upper = tf.matrix_band_part(Kw_xx, 0, -1) - tf.matrix_band_part(Kw_xx, 0, 0)  # <-- this diag part is incorrectly specified, should not be 1, 0; but rather 0, 0

    p1_weights_xy = p1_weights[:batch_size, batch_size:]  # Same.
    p1_weights_xy_normed = p1_weights_xy / tf.reduce_sum(p1_weights_xy)  # Same.
    Kw_xy = K_xy * p1_weights_xy_normed

    if weighted:
        # TODO: Sort why theory_unbiased and theory_upper don't work.
        option = 'theory_unbiased'

        if option == 'theory_unbiased':
            mmd = (tf.reduce_sum(Kw_xx_offdiag) / (batch_size * batch_size - 1) +
                   tf.reduce_sum(K_yy_offdiag) / (batch_size * batch_size - 1) -
                   2 * tf.reduce_sum(Kw_xy) / (batch_size * batch_size))
        elif option == 'theory_upper':
            mmd = (tf.reduce_sum(Kw_xx_upper) / num_combos_xx +
                   tf.reduce_sum(K_yy_upper) / num_combos_yy -
                   2 * tf.reduce_sum(Kw_xy) / (batch_size * batch_size))
        elif option == 'theory_biased':
            mmd = (tf.reduce_sum(Kw_xx) / (batch_size ** 2) +
                   tf.reduce_sum(K_yy) / (batch_size ** 2) -
                   2 * tf.reduce_sum(Kw_xy) / (batch_size ** 2))
        elif option == 'old':
            mmd = (tf.reduce_sum(Kw_xx_upper) +
                   tf.reduce_sum(K_yy_upper) / num_combos_yy -
                   2 * tf.reduce_sum(Kw_xy))

    else:
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_xx +
               tf.reduce_sum(K_yy_upper) / num_combos_yy -
               2 * tf.reduce_sum(K_xy) / (batch_size * batch_size))
    return mmd, exp_object, p1p2_weights_xx


# Beginning of graph.
z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')

g, g_vars = generator(z, reuse=False)
d_real, d_logit_real, d_vars = discriminator(x, reuse=False)
d_fake, d_logit_fake, _ = discriminator(g, reuse=True)

z_sample =  tf.placeholder(tf.float32, shape=[None, z_dim], name='z_sample')
g_sample, _ = generator(z_sample, reuse=True)
x_disc = tf.placeholder(tf.float32, shape=[None, x_dim], name='x_disc')
d_disc, _, _ = discriminator(x_disc, reuse=True)

# Define losses.
errors_real = sigmoid_cross_entropy_with_logits(d_logit_real,
    tf.ones_like(d_logit_real))
errors_fake = sigmoid_cross_entropy_with_logits(d_logit_fake,
    tf.zeros_like(d_logit_fake))
if weighted:
    weights_x = 1. / thinning_fn(x)
    weights_x_sum_normalized = weights_x / tf.reduce_sum(weights_x)
    d_loss_real = tf.reduce_mean(weights_x_sum_normalized * errors_real)
else:
    d_loss_real = tf.reduce_mean(errors_real)
d_loss_fake = tf.reduce_mean(errors_fake)
mmd, exp_object_eval, p1p2_weights_xx_eval = compute_mmd(x, g, batch_size, weighted=weighted)

d_loss = d_loss_real + d_loss_fake
if do_mmd:
    g_loss = mmd
else:
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

# Set optim nodes.
clip = 1
if clip:
    d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    d_grads_, d_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=d_vars))
    d_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in d_grads_])
    d_optim = d_opt.apply_gradients(zip(d_grads_clipped_, d_vars_))
    
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_grads_, g_vars_ = zip(*g_opt.compute_gradients(g_loss, var_list=g_vars))
    g_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in g_grads_])
    g_optim = g_opt.apply_gradients(zip(g_grads_clipped_, g_vars_))
else:
    d_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        g_loss, var_list=g_vars)
# End: Build model.
################################################################################


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
for it in range(500000):
    x_batch = sample_data(data_normed, batch_size)
    z_batch = get_sample_z(batch_size, z_dim)

    if not do_mmd:
        for _ in range(5):
            _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                    [d_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
                feed_dict={
                    x: x_batch,
                    z: z_batch})
        for _ in range(1):
            _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                    [g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
                feed_dict={
                    x: x_batch,
                    z: z_batch})
    else:
        _, g_loss_, eo, pp = sess.run([g_optim, g_loss, exp_object_eval, p1p2_weights_xx_eval],
            feed_dict={
                x: x_batch,
                z: z_batch})
        testing = sample_generator(1)
        if np.isnan(testing[0][0]):
            pdb.set_trace()

    if it % log_iter == 0:
        print("#################")
        print('Iter: {}, lr={}'.format(it, learning_rate))
        if not do_mmd:
            print('  d_logit_real: {}'.format(d_logit_real_[:5]))
            print('  d_logit_fake: {}'.format(d_logit_fake_[:5]))
            print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))

        n_sample = 1000
        generated = sample_generator(n_sample)

        # Print diagnostics.
        print(data_raw[np.random.choice(data_num, 5), :])
        print
        print(generated[:5])

        fig = plot(generated, data_raw, data_raw_unthinned, it)

        # Diagnostics for thinning_fn.
        thin_diag = 0
        if thin_diag:
            errors_real_, weights_x_, d_loss_real_ = sess.run(
                [errors_real, weights_x, d_loss_real],
                feed_dict={
                    x: x_batch[:5],
                    y: y_batch[:5]})
            print
            print(x_batch[:5] * data_raw_std[0] + data_raw_mean[0])
            print(errors_real_)
            print(weights_x_)
            print(d_loss_real_)
            pdb.set_trace()
