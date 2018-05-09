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
args = parser.parse_args()
tag = args.tag
weighted = args.weighted
do_p = args.do_p
data_num = 10000
batch_size = 256 
z_dim = 5
x_dim = 1
h_dim = 3
learning_rate = 1e-3
log_iter = 1000
log_dir = 'results/mmdgan1d_{}'.format(tag)


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


def sample_data(data, batch_size, weighted=weighted):
    assert data.shape[1] == 1, 'data shape not 1'
    idxs = np.random.choice(data_num, batch_size)
    batch_x = np.reshape(data[idxs, 0], [-1, 1])
    weights = None
    if weighted:
        weights = global_weights[idxs]
    return batch_x, weights


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

    plt.savefig('{}/{}.png'.format(log_dir, it))
    plt.close()


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def run_discrim(x_in):
    x_in = np.reshape(x_in, [-1, 1])
    return sess.run(d_disc, feed_dict={x_disc: x_in}) 


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


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


def compute_mmd(input1, input2, batch_size, w, weighted=False):
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
    K_yy_upper = upper(K_yy)

    p1_weights = tf.tile(w, [1, batch_size])
    p2_weights = tf.transpose(p1_weights)
    p1p2_weights = p1_weights * p2_weights
    p1p2_weights_upper = upper(p1p2_weights)
    p1p2_weights_upper_normed = p1p2_weights_upper / tf.reduce_sum(p1p2_weights_upper)
    p1_weights_normed = p1_weights / tf.reduce_sum(p1_weights)
    Kw_xx_upper = K_xx * p1p2_weights_upper_normed
    Kw_xy = K_xy * p1_weights_normed

    if weighted:
        mmd = (tf.reduce_sum(Kw_xx_upper) +
               tf.reduce_sum(K_yy_upper) / num_combos_yy -
               2 * tf.reduce_sum(Kw_xy))
    else:
        mmd = (tf.reduce_sum(K_xx_upper) / num_combos_xx +
               tf.reduce_sum(K_yy_upper) / num_combos_yy -
               2 * tf.reduce_sum(K_xy) / (batch_size * batch_size))
    return mmd


# Beginning of graph.
z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
w = tf.placeholder(tf.float32, shape=[batch_size, 1], name='w')
x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')

g, g_vars = generator(z, reuse=False)

z_sample =  tf.placeholder(tf.float32, shape=[None, z_dim], name='z_sample')
g_sample, _ = generator(z_sample, reuse=True)

# Define losses.
mmd = compute_mmd(x, g, batch_size, w, weighted=weighted)
g_loss = mmd

# Set optim nodes.
clip = 1
if clip:
    g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    g_grads_, g_vars_ = zip(*g_opt.compute_gradients(g_loss, var_list=g_vars))
    g_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in g_grads_])
    g_optim = g_opt.apply_gradients(zip(g_grads_clipped_, g_vars_))
else:
    g_optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        g_loss, var_list=g_vars)
# End: Build model.
################################################################################


# Load data.
data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned = \
    generate_data(data_num)
if do_p:
    data_normed = to_normed(data_raw_unthinned)
    data_raw = data_raw_unthinned

# Compute global approximation of weights (1/~T(x)).
global_weights = []
for v in data_normed:
    global_weights.append(1. / thinning_fn(to_raw(v)[0], is_tf=False))
global_weights = np.reshape(
    global_weights * data_num / np.sum(global_weights), [-1, 1])
    #global_weights / np.sum(global_weights), [-1, 1])

# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
for it in range(500000):
    x_batch, weights_batch = sample_data(data_normed, batch_size, weighted=weighted)
    z_batch = get_sample_z(batch_size, z_dim)

    if weighted:
        #_, g_loss_ = sess.run([g_optim, g_loss],
        _, g_loss_ = sess.run([g_optim, g_loss],
            feed_dict={
                x: x_batch,
                z: z_batch,
                w: weights_batch})
    else:
        _, g_loss_ = sess.run([g_optim, g_loss],
            feed_dict={
                x: x_batch,
                z: z_batch})

    if it % log_iter == 0:
        print("#################")
        print('Iter: {}, lr={}'.format(it, learning_rate))
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
