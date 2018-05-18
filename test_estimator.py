import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from scipy.stats import beta
from utils import compute_mmd


parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=int, default=1, choices=[1, 2, 3])
parser.add_argument('--data_mode', type=str, default='latent', choices=['latent', 'transformed'])
parser.add_argument('--alpha', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
args = parser.parse_args()
data_set = args.data_set
data_mode = args.data_mode
alpha = args.alpha
batch_size = args.batch_size


def compute_mmd_miw_numpy(arr1, arr2, arr1_weights):
    k1_in1, k2_in1, k3_in1, k4_in1 = np.split(arr1, 4)
    k1_in2, k2_in2, k3_in2, k4_in2 = np.split(arr2, 4)
    k1_in1_w, k2_in1_w, k3_in1_w, k4_in1_w = np.split(arr1_weights, 4)

    mmd1 = compute_mmd_iw_numpy(k1_in1, k1_in2, k1_in1_w)
    mmd2 = compute_mmd_iw_numpy(k2_in1, k2_in2, k2_in1_w)
    mmd3 = compute_mmd_iw_numpy(k3_in1, k3_in2, k3_in1_w)
    mmd4 = compute_mmd_iw_numpy(k4_in1, k4_in2, k4_in1_w)
    median_of_mmds = np.median([mmd1, mmd2, mmd3, mmd4])
    return median_of_mmds


def compute_mmd_iw_numpy(arr1, arr2, arr1_weights, sigma_list=None):
    """Computes mmd_iw between two numpy arrays, the first with weights."""
    if sigma_list is None:
        sigma_list = [0.1, 1.0, 10.0]

    n1 = len(arr1)
    n2 = len(arr2)
    num_combos_xx = n1 * (n1 - 1) / 2
    num_combos_yy = n2 * (n2 - 1) / 2

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
        #gamma = 1.0 / (2.0 * sigma**2)
        gamma = 1.0 / (2.0 * sigma)
        K += np.exp(-gamma * exp_object)
    K_xx = K[:n1, :n1]
    K_yy = K[n1:, n1:]
    K_xy = K[:n1, n1:]
    K_xx_upper = np.triu(K_xx)
    K_yy_upper = np.triu(K_yy)

    weights_tiled_horiz = np.tile(arr1_weights, [1, n1]) 
    p1_weights = weights_tiled_horiz
    p2_weights = np.transpose(p1_weights)
    p1p2_weights = p1_weights * p2_weights
    p1p2_weights_upper = np.triu(p1p2_weights)
    Kw_xx_upper = K_xx * p1p2_weights_upper
    Kw_xy = K_xy * p1_weights

    mmd = (np.sum(Kw_xx_upper) / num_combos_xx +
           np.sum(K_yy_upper) / num_combos_yy-
           2 * np.mean(Kw_xy))
    return mmd


def vert(arr):
    return np.reshape(arr, [-1, 1])


def sample(arr, n):
    sample_indices = np.random.choice(len(arr), n)
    return arr[sample_indices]


data_num = 10000
data_dim = 2
latent_dim = 10

beta_params = [1] * latent_dim
beta_params[0] = alpha

#############################################################
# BUILD DATA SET

if data_set == 1:
    # X ~ MP = Unif <-- Observed, thinned.
    # Y ~ P  = Beta <-- Target, unthinned.
    # Weights = 1/M = P/(MP) = Beta/Unif = Beta
    latent = np.random.uniform(0, 1, size=(data_num, latent_dim))
    latent_unthinned = np.random.uniform(0, 1, size=(data_num, latent_dim))
    #latent_unthinned = np.random.beta(beta_params, beta_params, (data_num, latent_dim))
    weights = vert(beta.pdf(latent[:, 0], alpha, 1.))
    weights_unthinned = vert(beta.pdf(latent_unthinned[:, 0], alpha, 1.))

    fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))
    data = np.dot(latent, fixed_transform)
    data_unthinned = np.dot(latent_unthinned, fixed_transform)

elif data_set == 2:
    # X ~ MP = Beta <-- Observed, thinned.
    # Y ~ P  = Unif <-- Target, unthinned.
    # Weights = 1/M = P/(MP) = Unif/Beta = 1/Beta
    latent = np.random.beta(beta_params, beta_params, (data_num, latent_dim))
    latent_unthinned = np.random.uniform(0, 1, size=(data_num, latent_dim))
    weights = vert(1. / beta.pdf(latent[:, 0], alpha, 1.))
    weights_unthinned = vert(1. / beta.pdf(latent_unthinned[:, 0], alpha, 1.))

    fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))
    data = np.dot(latent, fixed_transform)
    data_unthinned = np.dot(latent_unthinned, fixed_transform)

elif data_set == 3:
    # X ~ MP = Beta <-- Observed, thinned.
    # Y ~ P  = Unif <-- Target, unthinned.
    # Weights = 1/M = P/(MP) = Unif/Beta = 1/Beta
    latent = np.random.beta(beta_params, beta_params, (data_num, latent_dim))
    latent_unthinned = np.random.uniform(0, 1, size=(data_num, latent_dim))
    weights = vert(1. / beta.pdf(latent[:, 0], alpha, 1.))
    weights_unthinned = vert(1. / beta.pdf(latent_unthinned[:, 0], alpha, 1.))

    fixed_transform = np.random.normal(0, 1, size=(latent_dim, data_dim))
    data = np.dot(latent, fixed_transform)
    data_unthinned = np.dot(latent_unthinned, fixed_transform)




#############################################################
# DO COMPARISONS.
num_batches = 1000

# mode = ['latent', 'transformed']
mode = 'latent'

if mode == 'latent':
    observed = latent
    target = latent_unthinned
elif mode == 'transformed':
    observed = data
    target = data_unthinned

# Compute MMD_iw(X, Y) over B baches of batch_size samples.
mmd_iw = []
for i in range(num_batches):
    sample_indices = np.random.choice(data_num, batch_size)
    sample_observed = observed[sample_indices]
    sample_target = target[sample_indices]
    sample_observed_weights = weights[sample_indices]
    mmd_iw.append(
        compute_mmd_iw_numpy(sample_observed, sample_target, sample_observed_weights))
  
# Compute MMD_miw(X, Y) over B baches of batch_size samples.
mmd = []
mmd_miw = []
for i in range(num_batches):
    sample_indices = np.random.choice(data_num, batch_size)
    sample_observed = observed[sample_indices]
    sample_target = target[sample_indices]
    sample_observed_weights = weights[sample_indices]

    sample_mmd, _ = compute_mmd(sample_observed, sample_target)
    mmd.append(sample_mmd)
    mmd_miw.append(
        compute_mmd_miw_numpy(sample_observed, sample_target, sample_observed_weights))
  
plt.figure()
plt.hist([mmd, mmd_iw, mmd_miw], bins=30, label=['mmd', 'iw', 'miw'])
plt.legend()
plt.title('{}: {} batches of size {}, alpha={}'.format(
    data_mode, num_batches, batch_size, alpha))
file_name = 'mmd_iw_miw_on_{}.png'.format(data_mode)
plt.savefig(file_name)
plt.close()

iw = mmd_iw
miw = mmd_miw
print('Max IW: {}, Max MIW: {}'.format(max(np.abs(iw)), max(np.abs(miw))))
os.system('echo $PWD | mutt momod@utexas.edu -s "test_estimator"  -a "{}"'.format(file_name))


