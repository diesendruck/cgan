import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import scipy.stats as stats
import sys
sys.path.append('/home/maurice/mmd')
import tensorflow as tf
layers = tf.layers
import time

from matplotlib.gridspec import GridSpec
from tensorflow.examples.tutorials.mnist import input_data

from kl_estimators import naive_estimator as compute_kl
from mmd_utils import compute_mmd, compute_energy
from utils import get_data, generate_data, sample_data, dense, split_80_20


parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='test_0_0')
parser.add_argument('--data_dim', type=int, default=2)
parser.add_argument('--max_step', type=int, default=25000)
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--sampling', type=str, default='random',
                    choices=['random'])
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--noise_dim', type=int, default=10)
parser.add_argument('--h_dim', type=int, default=10)

args = parser.parse_args()
tag = args.tag
data_dim = args.data_dim
max_step = args.max_step
log_step = args.log_step
batch_size = args.batch_size
learning_rate_init = args.learning_rate
sampling = args.sampling
latent_dim = args.latent_dim
noise_dim = args.noise_dim
h_dim = args.h_dim

model_type = 'ce'
model_subtype, model_dim, model_runnum = tag.split('_')
model_dim = model_dim.replace('dim', '')
model_runnum = model_runnum.replace('run', '')
log_dir = 'results/{}_{}'.format(model_type, tag)


# Load data.
#(data_raw,
# data_raw_weights,
# data_raw_unthinned,
# data_raw_unthinned_weights,
# data_normed,
# data_raw_mean,
# data_raw_std) = generate_data(
#     data_num, data_dim, latent_dim, with_latents=False, m_weight=2.)  # m_weight since changed to 5.
(m_weight,
 data_raw,
 data_raw_weights,
 data_raw_unthinned,
 data_raw_unthinned_weights,
 data_normed,
 data_raw_mean,
 data_raw_std) = get_data(data_dim, with_latents=False)

# To do model selection, separate out two sets of unthinned data. Use the first
# to select the model, and the second to report that model's performance.
# Since data_raw_unthinned is sampled entirely separately from training data,
# and is not used in training, it will be used for validation and test sets.
(data_raw_unthinned_validation,
 data_raw_unthinned_test) = split_80_20(data_raw_unthinned)


def sigmoid_cross_entropy_with_logits(logits, labels):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, targets=labels)


def plot(generated, data_raw, data_raw_unthinned, log_dir, tag, step, measure_to_plot):
    gen_v1 = generated[:, 0] 
    gen_v2 = generated[:, 1] 
    raw_v1 = [d[0] for d in data_raw]
    raw_v2 = [d[1] for d in data_raw]
    raw_unthinned_v1 = [d[0] for d in data_raw_unthinned]
    raw_unthinned_v2 = [d[1] for d in data_raw_unthinned]

    # Evaluate D on grid.
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
    #ax_joint.imshow(vals_on_grid, interpolation='nearest', origin='lower',
    #    alpha=0.3, aspect='auto',
    #    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()])

    bins = np.arange(-5, 5, 0.2)
    ax_marg_x.hist([raw_v1, gen_v1], bins=bins, color=['gray', 'blue'],
        label=['data', 'gen'], alpha=0.3, normed=True)
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
    ax_raw_marg_y = fig.add_subplot(gs[5:8, 3], sharey=ax_raw)
    ax_raw.scatter(raw_unthinned_v1, raw_unthinned_v2, c='green', alpha=0.1)
    ax_raw_marg_x.hist([raw_unthinned_v1, gen_v1], bins=bins, color=['green', 'blue'],
        label=['validation', 'gen'], alpha=0.3, normed=True)
    ax_raw_marg_y.hist([raw_unthinned_v2, gen_v2], bins=bins, color=['green', 'blue'],
        label=['validation', 'gen'], alpha=0.3, normed=True, orientation='horizontal')
    ax_raw_marg_x.legend()
    ax_raw_marg_y.legend()
    plt.setp(ax_raw_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_raw_marg_y.get_yticklabels(), visible=False)
    ########

    plt.suptitle('{}. step: {}, discrepancy: {:.4f}'.format(
        log_dir[8:], step, measure_to_plot))
    plt.savefig('{}/{}.png'.format(log_dir, step))
    plt.close()


################################################################################
# BEGIN: Build model.
def Discriminator(inputs, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse) as d_vs:
        layer = dense(inputs, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=False)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=False,
            dropout=0.9)
        d_logit = dense(layer, 1, activation=None)
        d_prob = tf.nn.sigmoid(d_logit)
    d_vars = tf.contrib.framework.get_variables(d_vs)
    return d_prob, d_logit, d_vars 


def Generator(z, reuse=False):
    with tf.variable_scope('generator', reuse=reuse) as g_vs:
        layer = dense(z, h_dim, activation=tf.nn.elu)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=False)
        layer = dense(layer, h_dim, activation=tf.nn.elu, batch_residual=False,
            dropout=0.9)
        g = dense(layer, data_dim, activation=None)  # Outputing xy pairs.
    g_vars = tf.contrib.framework.get_variables(g_vs)
    return g, g_vars


def run_discrim(x_in):
    x_in = np.reshape(x_in, [-1, data_dim])
    return sess.run(d_real, feed_dict={x: x_in}) 


def get_sample_z(m, n):
    #return np.random.normal(0., 1., size=[m, n])
    tnorm = stats.truncnorm(-3, 3, loc=0, scale=1)
    return tnorm.rvs((m, n))


# Beginning of graph.
lr = tf.Variable(learning_rate_init, name='lr', trainable=False)
lr_update = tf.assign(lr, tf.maximum(lr * 0.5, 1e-8), name='lr_update')

z = tf.placeholder(tf.float32, shape=[None, noise_dim], name='z')
x = tf.placeholder(tf.float32, shape=[None, data_dim], name='x')

g, g_vars = Generator(z, reuse=False)
d_real, d_logit_real, d_vars = Discriminator(x, reuse=False)
d_fake, d_logit_fake, _ = Discriminator(g, reuse=True)

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
clip = 1
if clip:
    d_opt = tf.train.RMSPropOptimizer(learning_rate=lr)
    d_grads_, d_vars_ = zip(*d_opt.compute_gradients(d_loss, var_list=d_vars))
    d_grads_clipped_ = tuple(
        [tf.clip_by_value(grad, -0.01, 0.01) for grad in d_grads_])
    d_optim = d_opt.apply_gradients(zip(d_grads_clipped_, d_vars_))
else:
    d_optim = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
        d_loss, var_list=d_vars)
g_optim = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(
    g_loss, var_list=g_vars)
# End: Build model.
################################################################################


# Start session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# train()
# Start clock to time execution in chunks of log_step steps.
t0 = time.time()
for step in range(max_step):
    #####
    # BEGINNING OF TIMED SEGMENT

    x_batch_preup, w_batch_preup = sample_data(
        data_normed, data_raw_weights, batch_size)
    z_batch = get_sample_z(batch_size, noise_dim)

    # UPSAMPLE WITHIN BATCH.
    x_batch = x_batch_preup[:]
    if sampling == 'random':
        # Upsample then randomly select batch_size to use.
        #for x_, w_ in zip(x_batch_preup, w_batch_preup):
        #    k = int(round(w_))
        #    for _ in range(k - 1):
        #        x_batch.append(x_)
        #x_batch = np.reshape(x_batch, [-1, data_dim])
        #x_batch = x_batch[np.random.choice(len(x_batch), batch_size)]
        for x_, w_ in zip(x_batch_preup, w_batch_preup):
            k = int(round(w_))
            for _ in range(k - 1):
                x_batch = np.concatenate((x_batch, np.reshape(x_, [1, -1])), axis=0)
        x_batch = x_batch[np.random.choice(len(x_batch), batch_size)]


    fetch_dict = {
        z: z_batch,
        x: x_batch}

    for _ in range(5):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
            [d_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            fetch_dict)
    for _ in range(1):
        _, d_logit_real_, d_logit_fake_, d_loss_, g_loss_ = sess.run(
            [g_optim, d_logit_real, d_logit_fake, d_loss, g_loss],
            fetch_dict)

    if step % 100000 == 499:
        sess.run(lr_update)

    if step > 0 and step % log_step == 0:
        #####
        # END OF TIMED SEGMENT
        # Stop clock after log_step training steps.
        t1 = time.time()
        chunk_time = t1 - t0

        n_sample = 1000 
        z_sample = get_sample_z(n_sample, noise_dim)

        g_out = sess.run(g, feed_dict={z: z_sample})
        generated_normed = g_out
        generated = np.array(generated_normed) * data_raw_std + data_raw_mean

        nanlist = [[i, v] for i,v in enumerate(generated) if (np.isnan(v[0]) or np.isnan(v[1]))]
        if len(nanlist) > 0:
            print('#####################\nGENERATED NANs\n####################')


        ###### LOG VALIDATION AND TEST SCORES #####
	def compute_discrepancies(cand_set, ref_set):
	    """Computes discrepancies between two sets."""
	    cand_set_n = len(cand_set)
	    ref_set_n = len(ref_set)
	    n_sample = 1000
	    assert ((cand_set_n >= n_sample) & (ref_set_n >= n_sample)), \
		'n_sample too high for one of the inputs.'
	    # Compute MMD, Energy, and KL, between simulations and unthinned data.
	    mmd_, _ = compute_mmd(
		cand_set[np.random.choice(cand_set_n, n_sample)],
		ref_set[np.random.choice(ref_set_n, n_sample)])
	    energy_ = compute_energy(
		cand_set[np.random.choice(cand_set_n, n_sample)],
		ref_set[np.random.choice(ref_set_n, n_sample)])
	    kl_ = compute_kl(
		cand_set[np.random.choice(cand_set_n, n_sample)],
		ref_set[np.random.choice(ref_set_n, n_sample)],
		k=5)
	    return mmd_, energy_, kl_


        d_nolatent = data_raw[:, -data_dim:]
        d_nolatent_v = data_raw_unthinned_validation[:, -data_dim:]
        d_nolatent_t = data_raw_unthinned_test[:, -data_dim:]
        g_nolatent = generated[:, -data_dim:]

        (mmd_gen_vs_unthinned_validation,
         energy_gen_vs_unthinned_validation,
         kl_gen_vs_unthinned_validation) = compute_discrepancies(g_nolatent, d_nolatent_v)
        (mmd_gen_vs_unthinned_test,
         energy_gen_vs_unthinned_test,
         kl_gen_vs_unthinned_test) = compute_discrepancies(g_nolatent, d_nolatent_t)
        ############################################


        if data_dim == 2:
            measure_to_plot = mmd_gen_vs_unthinned_validation
            fig = plot(g_nolatent, d_nolatent, d_nolatent_v, log_dir, tag,
                step, measure_to_plot)

        if np.isnan(d_loss_):
            sys.exit('got nan')

        # Print diagnostics.
        print("#################")
        lr_ = sess.run(lr)
        print('{}_{}'.format(model_type, tag))
        print('Iter: {}, lr={}'.format(step, lr_))
        print('  d_loss: {:.4}'.format(d_loss_))
        print('  g_loss: {:.4}'.format(g_loss_))
        print('  mmd_gen_vs_unthinned: {:.4}'.format(mmd_gen_vs_unthinned_validation))
        with open(os.path.join(log_dir, 'scores_mmd.txt'), 'a') as f:
            f.write(str(mmd_gen_vs_unthinned_validation)+'\n')
        with open(os.path.join(log_dir, 'scores_energy.txt'), 'a') as f:
            f.write(str(energy_gen_vs_unthinned_validation)+'\n')
        with open(os.path.join(log_dir, 'scores_kl.txt'), 'a') as f:
            f.write(str(kl_gen_vs_unthinned_validation)+'\n')


        # Plot timing and performance together.
        with open(os.path.join(log_dir, 'perf.txt'), 'a') as f:
            f.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                model_type, model_subtype, model_dim, model_runnum, step,
                g_loss_,
                mmd_gen_vs_unthinned_validation,
                energy_gen_vs_unthinned_validation,
                kl_gen_vs_unthinned_validation,
                mmd_gen_vs_unthinned_test,
                energy_gen_vs_unthinned_test,
                kl_gen_vs_unthinned_test,
                chunk_time))

        # Restart clock for next log_step training steps.
        t0 = time.time()
