import tensorflow as tf
layers = tf.layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
import seaborn as sns
from matplotlib.gridspec import GridSpec


mb_size = 128 
z_dim = 5  # Latent
x_dim = 1  # Label
y_dim = 1  # Data
h_dim = 5


def plot_2d(var1, var2, background_data, background_discrim=False, save_name=None):
    # Unnormalize data.
    v1 = np.array(var1) * data_raw_std[0] + data_raw_mean[0]
    v2 = np.array(var2) * data_raw_std[1] + data_raw_mean[1]

    fig = plt.figure()
    gs = GridSpec(4,4)
    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4,3], sharey=ax_joint)

    """
    if background_discrim:
        bd = np.array(background_data)
        grid1 = np.linspace(min(bd[:, 0], max(bd[:, 0]))
        grid2 = np.linspace(min(bd[:, 1], max(bd[:, 1]))
        plt.imshow(vals_on_grid, interpolation='nearest', aspect='equal',
            extent=[grid1.min(), grid1.max(), grid2.min(), grid2.max()])
        plt.colorbar()
    """
        
    bd = np.array(background_data)
    bd = bd * data_raw_std + data_raw_mean
    bd_v1 = [d[0] for d in bd]
    bd_v2 = [d[1] for d in bd]
    ax_joint.scatter(bd_v1, bd_v2, c='gray', alpha=0.3)
    ax_joint.scatter(v1, v2)
    #ax_marg_x.hist([bd_v1, v1], bins=30, color=['gray', 'blue'], label=['b', 'd'])
    #ax_marg_y.hist([bd_v2, v2], bins=30, color=['gray', 'blue'], label=['b', 'd'], orientation="horizontal")
    ax_marg_x.hist(v1, bins=30)
    ax_marg_y.hist(v2, bins=30, orientation="horizontal")

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel('Joint: height (ft)')
    ax_joint.set_ylabel('Joint: income ($)')

    # Set labels on marginals
    ax_marg_y.set_xlabel('Marginal: income')
    ax_marg_x.set_ylabel('Marginal: height')

    if save_name:
        plt.savefig(save_name)


def plot_1d(generated, data_raw, it):
    gen_v1 = generated[:, 0] 
    gen_v2 = generated[:, 1] 

    # Will use normalized data for evaluation of D.
    data_normed = (data_raw - data_raw_mean) / data_raw_std

    # Evaluate D on grid.
    grid_gran = 20
    grid1 = np.linspace(min(data_normed[:, 0]), max(data_normed[:, 0]), grid_gran)
    grid2 = np.linspace(min(data_normed[:, 1]), max(data_normed[:, 1]), grid_gran)
    vals_on_grid = np.zeros((grid_gran, grid_gran))
    for i in range(grid_gran):
        for j in range(grid_gran):
            vals_on_grid[i][j] = run_discrim(grid1[i], grid2[j])

    if 0:
        plt.figure()
        #plt.subplot(211)
        #plt.imshow(vals_on_grid, interpolation='nearest')
        #plt.colorbar()
        plt.subplot(211)
        plt.imshow(vals_on_grid, interpolation='nearest', origin='lower',
            extent=[grid1.min(), grid1.max(), grid2.min(), grid2.max()])
        plt.colorbar()
        plt.subplot(212)
        raw_v1 = [d[0] for d in data_raw]
        raw_v2 = [d[1] for d in data_raw]
        plt.scatter(raw_v1, raw_v2, c='gray', alpha=0.3)
        plt.scatter(gen_v1, gen_v2)
        if save_name:
            plt.savefig(save_name)
        return None

    if 1:
        fig = plt.figure()
        gs = GridSpec(4, 4)
        ax_joint = fig.add_subplot(gs[1:4, 0:3])
        ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
        ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

        raw_v1 = [d[0] for d in data_raw]
        raw_v2 = [d[1] for d in data_raw]
        ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.3)
        ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
        ax_marg_x.hist([raw_v1, gen_v1], bins=30, color=['gray', 'blue'], label=['d', 'g'], alpha=0.3, normed=True)
        ax_marg_y.hist([raw_v2, gen_v2], bins=30, color=['gray', 'blue'], label=['d', 'g'], orientation="horizontal", alpha=0.3, normed=True)
        ax_marg_x.legend()
        ax_marg_y.legend()

        # Turn off tick labels on marginals
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Set labels on joint
        ax_joint.set_xlabel('Joint: height (ft)')
        ax_joint.set_ylabel('Joint: income ($)')

        # Set labels on marginals
        ax_marg_y.set_xlabel('Marginal: income')
        ax_marg_x.set_ylabel('Marginal: height')

        plt.savefig('iwgan_out/{}.png'.format(it))
        plt.close()

        # Also plot heatmap.
        plt.figure()
        plt.imshow(vals_on_grid, interpolation='nearest', aspect='equal',
            extent=[grid1.min(), grid1.max(), grid2.min(), grid2.max()])
        plt.colorbar()
        plt.savefig('iwgan_out/heatmap.png')
        plt.close()


def generate_data(n):
    ##latent = np.random.normal(10., 1., n)
    #latent = np.random.choice([10., 20.], n)
    #v1 = np.zeros(n)  # Height
    #v2 = np.zeros(n)  # Income
    #for i in range(n):
    #    v1[i] = np.random.normal(latent[i], 1.)  # Arbitrary v1.
    #    v2[i] = np.random.normal(latent[i], 1.)  # Arbitrary v2.
    #data = np.hstack((
    #    np.reshape(v1, [-1, 1]),
    #    np.reshape(v2, [-1, 1]),
    #    np.reshape(latent, [-1, 1])))
    #return data 

    # Consider a latent variable that regulates height and income.
    latent = np.random.gamma(4., 8., n)
    v1 = np.zeros(n)  # Height
    v2 = np.zeros(n)  # Income
    for i in range(n):
        v1_mean = -0.00003 * np.exp(-1.0 * (0.13 * latent[i] - 12)) + 5.5
        v1[i] = np.random.normal(v1_mean, 0.1)  # Height
        #v2[i] = np.random.exponential(latent[i] / 50 * 50000)  # Income
        v2[i] = latent[i] / 50 * 50000 + \
            abs(np.random.normal(0, 9999. / (1. + np.exp(0.15*(latent[i] - 40.))) + 1.))  # Income
        #v2[i] = abs(np.random.normal(2500 * (0.12 * latent[i]) ** 2.2, abs(-0.1 * latent[i] + 6) * 5000))
    data_raw = np.hstack((
        np.reshape(v1, [-1, 1]),
        np.reshape(v2, [-1, 1]),
        np.reshape(latent, [-1, 1])))
    data_raw_mean = np.mean(data_raw, axis=0)
    data_raw_std = np.std(data_raw, axis=0)
    data = (data_raw - data_raw_mean) / data_raw_std 
    return data, data_raw, data_raw_mean, data_raw_std 


def sample_data(batch_size):
    assert data.shape[1] == 3, 'data shape not 3'
    idxs = np.random.choice(data_num, batch_size)
    batch_x = np.reshape(data[idxs, 0], [-1, 1])
    batch_y = np.reshape(data[idxs, 1], [-1, 1])
    return batch_x, batch_y


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)

# Load data.
data_num = 10000
data, data_raw, data_raw_mean, data_raw_std = generate_data(data_num)

################################################################################
# BEGIN: Build model.
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def dense(x, width, activation, batch_residual=False):
    if not batch_residual:
        x_ = layers.dense(x, width, activation=activation)
        return layers.batch_normalization(x_)
    else:
        x_ = layers.dense(x, width, activation=activation, use_bias=False)
        return layers.batch_normalization(x_) + x


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def generator(z, reuse=False):
    #inputs = tf.concat(axis=1, values=[z, x])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(z, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, x_dim + y_dim, activation=None)  # Outputing xy pairs.
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def thinning_fn(inputs):
    """In 2D case, thinning based on x only. Inputs is a vector of x values."""
    return 0.99 / (1. + tf.exp(-0.95 * (inputs - 3.))) + 0.01


# Beginning of graph.
z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
x = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
y = tf.placeholder(tf.float32, shape=[None, y_dim], name='y')
real = tf.concat([x, y], axis=1)

g, g_vars = generator(z, reuse=False)
d_real, d_logit_real, d_vars = discriminator(real, reuse=False)
d_fake, d_logit_fake, _ = discriminator(g, reuse=True)

errors_real = sigmoid_cross_entropy_with_logits(d_logit_real,
    tf.ones_like(d_logit_real))
errors_fake = sigmoid_cross_entropy_with_logits(d_logit_fake,
    tf.zeros_like(d_logit_fake))
weights_x = 1. / thinning_fn(x)
d_loss_real = tf.reduce_mean(weights_x * errors_real)
d_loss_fake = tf.reduce_mean(errors_fake)

# Assemble losses.
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

# Set optim nodes.
clip = 0
if clip:
    d_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    d_grads_, d_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=d_vars))
    d_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in d_grads_])
    d_optim = d_opt.apply_gradients(zip(d_grads_clipped_, d_vars_))
else:
    d_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
        d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    g_loss, var_list=g_vars)

# End: Build model.
################################################################################


def run_discrim(x_in, y_in):
    x_in = np.reshape(x_in, [-1, 1])
    y_in = np.reshape(y_in, [-1, 1])
    return sess.run(d_real, feed_dict={x: x_in, y: y_in}) 


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_iter = 2000

if not os.path.exists('iwgan_out/'):
    os.makedirs('iwgan_out/')

# train()
for it in range(50000):
    x_batch, y_batch = sample_data(mb_size)
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
        print("#################")
        print('Iter: {}'.format(it))
        print('  d_logit_real: {}'.format(d_logit_real_[:5]))
        print('  d_logit_fake: {}'.format(d_logit_fake_[:5]))
        print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))

        n_sample = 1000
        z_sample = get_sample_z(n_sample, z_dim)
        g_out = sess.run(g, feed_dict={z: z_sample})
        generated = np.array(g_out) * data_raw_std[:2] + data_raw_mean[:2]

        # Print diagnostics.
        print(data_raw[np.random.choice(data_num, 5), :])
        print
        print(generated[:5])

        #fig = plot_2d([p[0] for p in g_out], [p[1] for p in g_out], data,
        #    save_name='out/{}.png'.format(str(it)))
        fig = plot_1d(generated, data_raw, it)

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
