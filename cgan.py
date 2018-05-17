import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sys
import tensorflow as tf
layers = tf.layers

from tensorflow.examples.tutorials.mnist import input_data
from utils import generate_data, thinning_fn, sample_data, compute_mmd 


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='test')
args = parser.parse_args()
tag = args.tag

data_num = 10000
data_dim = 2
latent_dim = 5

batch_size = 1024
noise_dim = 10
h_dim = 5
learning_rate = 1e-4
log_iter = 1000
log_dir = 'results/cgan_{}'.format(tag)
max_iter = 100000


# Load data.
(data_raw,
 data_raw_weights,
 data_raw_unthinned,
 data_raw_unthinned_weights,
 data_normed,
 data_raw_mean,
 data_raw_std) = generate_data(data_num, data_dim, latent_dim, with_latents=True)


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def plot(generated, data_raw, data_raw_unthinned, it, mmd_gen_vs_unthinned):
    gen_v1 = generated[:, latent_dim + 0]
    gen_v2 = generated[:, latent_dim + 1]
    raw_v1 = [d[latent_dim + 0] for d in data_raw]
    raw_v2 = [d[latent_dim + 1] for d in data_raw]
    raw_unthinned_v1 = [d[latent_dim + 0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[latent_dim + 1] for d in data_raw_unthinned]

    # Evaluate D on grid.
    grid_gran = 20
    grid_x = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)
    grid_y = np.linspace(min(data_raw[:, 1]), max(data_raw[:, 1]), grid_gran)
    #vals_on_grid = np.zeros((grid_gran, grid_gran))
    #for i in range(grid_gran):
    #    for j in range(grid_gran):
    #        grid_x_normed = (grid_x[i] - data_raw_mean[0]) / data_raw_std[0]
    #        grid_y_normed = (grid_y[j] - data_raw_mean[0]) / data_raw_std[0]
    #        vals_on_grid[i][j] = run_discrim(grid_x_normed, grid_y_normed)

    fig = plt.figure()
    gs = GridSpec(8, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.1)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    ax_joint.set_aspect('auto')
    #ax_joint.imshow(vals_on_grid, interpolation='nearest', origin='lower', alpha=0.3, aspect='auto',
    #    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])
    #ax_thinning = ax_joint.twinx()
    #ax_thinning.plot(grid_x, thinning_fn(grid_x, is_tf=False), color='red', alpha=0.3)
    ax_marg_x.hist([raw_v1, gen_v1], bins=30, color=['gray', 'blue'],
        label=['data', 'gen'], alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=30, color=['gray', 'blue'],
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


def discriminator(latent, x, reuse=False):
    inputs = tf.concat(axis=1, values=[latent, x])
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def generator(z, latent, reuse=False):
    inputs = tf.concat(axis=1, values=[z, latent])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, data_dim, activation=None)
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def run_discrim(x_in, y_in):
    x_in = np.reshape(x_in, [-1, 1])
    y_in = np.reshape(y_in, [-1, 1])
    return sess.run(d_real, feed_dict={x: x_in, latent: y_in}) 


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


# Beginning of graph.
z = tf.placeholder(tf.float32, shape=[None, noise_dim], name='z')
x = tf.placeholder(tf.float32, shape=[None, data_dim], name='x')
latent = tf.placeholder(tf.float32, shape=[None, latent_dim], name='latent')

g, g_vars = generator(z, latent, reuse=False)  # Takes in noise and label.
d_real, d_logit_real, d_vars = discriminator(latent, x, reuse=False)  # Takes in label and data.
d_fake, d_logit_fake, _ = discriminator(latent, g, reuse=True)  # Takes in label and data.

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


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
for it in range(max_iter):
    d_batch, w_batch = sample_data(data_normed, data_raw_weights, batch_size)
    latent_batch, x_batch = d_batch[:, :latent_dim], d_batch[:, -data_dim:]

    z_batch = get_sample_z(batch_size, noise_dim)

    for _ in range(5):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                [d_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            feed_dict={
                z: z_batch,
                x: x_batch,
                latent: latent_batch})
    for _ in range(1):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
                [g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            feed_dict={
                z: z_batch,
                x: x_batch,
                latent: latent_batch})

    if it % log_iter == 0:
        n_sample = 10000
        z_sample = get_sample_z(n_sample, noise_dim)

        # SAMPLE FROM CONDITIONAL, REGULATED BY THINNING_FN.
        raw_marginal = data_raw[:, :latent_dim]
        #thinning_fn_values = thinning_fn(raw_marginal, is_tf=False)
        thinning_fn_values = np.zeros((data_num, 1))
        for i in range(data_num):
            thinning_fn_values[i] = thinning_fn(raw_marginal[i], is_tf=False)
        weights = 1. / thinning_fn_values
        weights_sum_normalized = weights / np.sum(weights)
        sample_indices = np.random.choice(data_num, size=n_sample,
            p=weights_sum_normalized.flatten())
        latent_sample_unnormed = raw_marginal[sample_indices]
        latent_sample = (
            (latent_sample_unnormed - data_raw_mean[:latent_dim]) / 
             data_raw_std[:latent_dim])

        g_out = sess.run(g, feed_dict={z: z_sample, latent: latent_sample})
        generated_normed = np.hstack((latent_sample, g_out))
        generated = np.array(generated_normed) * data_raw_std + data_raw_mean

        # Compute MMD only between data dimensions, and not latent ones.
        mmd_gen_vs_unthinned, _ = compute_mmd(
            generated[np.random.choice(n_sample, 500), -data_dim:],
            data_raw_unthinned[np.random.choice(data_num, 500), -data_dim:])

        if data_dim == 2:
            fig = plot(generated, data_raw, data_raw_unthinned, it,
                mmd_gen_vs_unthinned)

        if np.isnan(d_loss_):
            sys.exit('got nan')

        # Print diagnostics.
        print("#################")
        print('Iter: {}'.format(it))
        print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))
        print('  mmd_gen_vs_unthinned: {:.4}'.format(mmd_gen_vs_unthinned))
        print(data_raw[np.random.choice(data_num, 1), :5])
        print
        print(generated[:1, :5])
        with open(os.path.join(log_dir, 'scores.txt'), 'a') as f:
            f.write(str(mmd_gen_vs_unthinned)+'\n')
