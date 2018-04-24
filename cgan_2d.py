import tensorflow as tf
layers = tf.layers
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from matplotlib.gridspec import GridSpec


mb_size = 640
z_dim = 2
X_dim = 2
y_dim = 1
h_dim = 2


def plot_2d(var1, var2, save_name=None, background_data=None):
    fig = plt.figure()

    gs = GridSpec(4,4)

    ax_joint = fig.add_subplot(gs[1:4,0:3])
    ax_marg_x = fig.add_subplot(gs[0,0:3])
    ax_marg_y = fig.add_subplot(gs[1:4,3])

    if background_data is not None:
        bd = background_data
        ax_joint.scatter([d[0] for d in bd], [d[1] for d in bd], c='gray', alpha=0.3)
    ax_joint.scatter(var1, var2)
    ax_marg_x.hist(var1, bins=30)
    ax_marg_y.hist(var2, bins=30, orientation="horizontal")

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


def generate_data(n):
    #latent = np.random.normal(10., 1., n)
    latent = np.random.choice([10., 20.], n)
    v1 = np.zeros(n)  # Height
    v2 = np.zeros(n)  # Income
    for i in range(n):
        v1[i] = np.random.normal(latent[i], 1.)  # Height
        v2[i] = np.random.normal(latent[i], 1.)  # Height
    plot_2d(v1, v2, save_name='test.png')
    data = np.hstack((
        np.reshape(v1, [-1, 1]),
        np.reshape(v2, [-1, 1]),
        np.reshape(latent, [-1, 1])))
    return data 

    ## Consider a latent variable that regulates height and income.
    #latent = np.random.gamma(2, 20, n)
    #v1 = np.zeros(n)  # Height
    #v2 = np.zeros(n)  # Income
    #for i in range(n):
    #    v1_mean = -0.00003 * np.exp(-1.0 * (0.13 * latent[i] - 12)) + 5.5
    #    v1[i] = np.random.normal(v1_mean, 0.1)  # Height
    #    #v2[i] = np.random.exponential(latent[i] / 50 * 50000)  # Income
    #    v2[i] = np.random.normal(latent[i] * 100 + 50000)  # Income
    #plot_2d(v1, v2, save_name='test.png')
    #data = np.hstack((
    #    np.reshape(v1, [-1, 1]),
    #    np.reshape(v2, [-1, 1]),
    #    np.reshape(latent, [-1, 1])))
    #return data 


def sample_data(batch_size):
    assert data.shape[1] == 3, 'data shape not 3'
    idxs = np.random.choice(data_num, batch_size)
    batch_X = data[idxs, :2]
    batch_y = np.reshape(data[idxs, 2], [-1, 1])
    return batch_X, batch_y


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)

# Load data.
data_num = 10000
data = generate_data(data_num)
input_dim = 2


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


def discriminator(x, y, reuse=False):
    inputs = tf.concat(axis=1, values=[x, y])
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def generator(z, y, reuse=False):
    inputs = tf.concat(axis=1, values=[z, y])
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        g = dense(layer, X_dim, activation=None)
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])

X = tf.placeholder(tf.float32, shape=[None, input_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

g, g_vars = generator(z, y, reuse=False)
d_real, d_logit_real, d_vars = discriminator(X, y, reuse=False)
d_fake, d_logit_fake, _ = discriminator(g, y, reuse=True)

d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
    d_logit_real, tf.ones_like(d_logit_real)))
d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(
    d_logit_fake, tf.zeros_like(d_logit_fake)))

# Assemble losses.
d_loss = d_loss_real + d_loss_fake
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))

# Set optim nodes.
d_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
    g_loss, var_list=g_vars)

# End: Build model.
################################################################################


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

log_iter = 2000

if not os.path.exists('out/'):
    os.makedirs('out/')

for it in range(1000000):
    X_batch, y_batch = sample_data(mb_size)
    z_batch = get_sample_z(mb_size, z_dim)

    _, _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
            [d_optim, g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
        feed_dict={
            X: X_batch,
            z: z_batch,
            y: y_batch})

    if it % log_iter == 0:
        print("#################")
        print('Iter: {}'.format(it))
        print('  d_logit_real: {}'.format(d_logit_real_[:10]))
        print('  d_logit_fake: {}'.format(d_logit_fake_[:10]))
        print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))

        n_sample = 10
        z_sample = get_sample_z(n_sample, z_dim)
        #y_sample = np.reshape(np.random.uniform(1, 90, n_sample), [-1, 1])
        y_sample = np.reshape(np.random.choice([10., 20.], n_sample), [-1, 1])
        g_out = sess.run(g, feed_dict={z: z_sample, y: y_sample})

        print(data[np.random.choice(data_num, 10), :])
        print
        print(np.hstack((g_out, y_sample)))

        fig = plot_2d([p[0] for p in g_out], [p[1] for p in g_out],
            save_name='out/{}.png'.format(str(it)), background_data=data)
