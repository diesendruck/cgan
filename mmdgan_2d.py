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
                    help='Chooses whether to use weighted MMD.')
parser.add_argument('--do_p', default=False, action='store_true', dest='do_p',
                    help='Choose whether to use P, instead of TP')
args = parser.parse_args()
tag = args.tag
weighted = args.weighted
do_p = args.do_p
data_num = 10000
batch_size = 256 
z_dim = 5  # Latent (Age)
x_dim = 1  # Label (Height)
y_dim = 1  # Data (Income)
h_dim = 5
learning_rate_init = 1e-2
log_iter = 1000
log_dir = 'mmdgan_out_{}'.format(tag)


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
        v2 = np.random.normal(0., 2. - v1 / 3.)
        #v2 = np.random.normal(0.,  v1 / 6.)
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

    sampling_fn = gen_from_angled_bar

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


def thinning_fn(inputs, is_tf=True):
    """Thinning on x only (height). Inputs is a vector of x values."""
    if is_tf:
        return 0.9 / (1. + tf.exp(-0.95 * (inputs - 3.))) + 0.1
    else:
        return 0.9 / (1. + np.exp(-0.95 * (inputs - 3.))) + 0.1


# Load data.
data_normed, data_raw, data_raw_mean, data_raw_std, data_raw_unthinned = \
    generate_data(data_num)
if do_p:
    data_normed = to_normed(data_raw_unthinned)
    data_raw = data_raw_unthinned


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


def plot(generated, data_raw, it):
    gen_v1 = generated[:, 0] 
    gen_v2 = generated[:, 1] 

    # Will use normalized data for evaluation of D.
    data_normed = to_normed(data_raw)

    # Evaluate D on grid.
    grid_gran = 20
    grid1 = np.linspace(min(data_raw[:, 0]), max(data_raw[:, 0]), grid_gran)

    fig = plt.figure()
    gs = GridSpec(4, 4)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    ax_joint.scatter(raw_v1, raw_v2, c='gray', alpha=0.3)
    ax_joint.scatter(gen_v1, gen_v2, alpha=0.3)
    ax_joint.set_aspect('auto')
    ax_thinning = ax_joint.twinx()
    ax_thinning.plot(grid1, thinning_fn(grid1, is_tf=False), color='red', alpha=0.3)
    ax_marg_x.hist([raw_v1, gen_v1], bins=30, color=['gray', 'blue'],
        label=['d', 'g'], alpha=0.3, normed=True)
    ax_marg_y.hist([raw_v2, gen_v2], bins=30, color=['gray', 'blue'],
        label=['d', 'g'], alpha=0.3, normed=True, orientation="horizontal",)
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

    plt.savefig('{}/{}.png'.format(log_dir, it))
    plt.close()


def get_sample_z(m, n):
    return np.random.normal(0., 1., size=[m, n])


def upper(mat):
    return tf.matrix_band_part(mat, 0, -1) - tf.matrix_band_part(mat, 0, 0)


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


def discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
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
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=True)
        g = dense(layer, x_dim + y_dim, activation=None)  # Outputing xy pairs.
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def compute_mmd(input1, input2, weighted=False):
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
    K_xx_upper = upper(K_xx)
    K_yy_upper = upper(K_yy)

    x_unnormed = v[:batch_size, :1]
    weights_x = 1. / thinning_fn(x_unnormed)
    weights_x_tiled_horiz = tf.tile(weights_x, [1, batch_size])
    p1_weights = weights_x_tiled_horiz
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
lr = tf.Variable(learning_rate_init, name='lr', trainable=False)
lr_update = tf.assign(lr, tf.maximum(lr * 0.5, 1e-8), name='lr_update')

z = tf.placeholder(tf.float32, shape=[batch_size, z_dim], name='z')
z_sample = tf.placeholder(tf.float32, shape=[None, z_dim], name='z_sample')
x = tf.placeholder(tf.float32, shape=[batch_size, x_dim], name='x')
y = tf.placeholder(tf.float32, shape=[batch_size, y_dim], name='y')
real = tf.concat([x, y], axis=1)

g, g_vars = generator(z, reuse=False)
g_sample, _ = generator(z_sample, reuse=True)
d_real, d_logit_real, d_vars = discriminator(real, reuse=False)
d_fake, d_logit_fake, _ = discriminator(g, reuse=True)

# Define losses.
mmd = compute_mmd(real, g, weighted=weighted)
errors_real = sigmoid_cross_entropy_with_logits(d_logit_real,
    tf.ones_like(d_logit_real))
errors_fake = sigmoid_cross_entropy_with_logits(d_logit_fake,
    tf.zeros_like(d_logit_fake))
if weighted:
    x_unnormed = x * data_raw_std[0] + data_raw_mean[0]
    weights_x = 1. / thinning_fn(x_unnormed)
    weights_x_sum_normalized = weights_x / tf.reduce_sum(weights_x)
    d_loss_real = tf.reduce_sum(weights_x_sum_normalized * errors_real)
else:
    d_loss_real = tf.reduce_mean(errors_real)
d_loss_fake = tf.reduce_mean(errors_fake)


# Assemble final losses.
#d_loss = d_loss_real + d_loss_fake
#g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#    logits=d_logit_fake, labels=tf.ones_like(d_logit_fake)))
g_loss = mmd

# Set optim nodes.
#clip = 0
#if clip:
#    d_opt = tf.train.AdamOptimizer(learning_rate=1e-4)
#    d_grads_, d_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=d_vars))
#    d_grads_clipped_ = tuple(
#        [tf.clip_by_value(grad, -0.01, 0.01) for grad in d_grads_])
#    d_optim = d_opt.apply_gradients(zip(d_grads_clipped_, d_vars_))
#else:
#    d_optim = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(
#        d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(
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
    x_batch, y_batch = sample_data(data_normed, batch_size)
    z_batch = get_sample_z(batch_size, z_dim)

    #for _ in range(5):
    #    _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
    #            [d_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
    #        feed_dict={
    #            x: x_batch,
    #            z: z_batch,
    #            y: y_batch})
    #for _ in range(1):
    #    _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
    #            [g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
    #        feed_dict={
    #            x: x_batch,
    #            z: z_batch,
    #            y: y_batch})
    _, g_loss_ = sess.run(
            [g_optim, g_loss],
        feed_dict={
            x: x_batch,
            z: z_batch,
            y: y_batch})

    if it % 10000 == 9999:
        sess.run(lr_update)

    if it % log_iter == 0:
        print("#################")
        lr_ = sess.run(lr)
        print('Iter: {}, lr={:.4f}'.format(it, lr_))
        #print('  d_logit_real: {}'.format(d_logit_real_[:5]))
        #print('  d_logit_fake: {}'.format(d_logit_fake_[:5]))
        #print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))

        n_sample = 1000 
        z_sample_input = get_sample_z(n_sample, z_dim)
        g_out = sess.run(g_sample, feed_dict={z_sample: z_sample_input})
        generated = np.array(g_out) * data_raw_std[:2] + data_raw_mean[:2]

        # Print diagnostics.
        print(data_raw[np.random.choice(data_num, 5), :])
        print
        print(generated[:5])

        fig = plot(generated, data_raw, it)

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
