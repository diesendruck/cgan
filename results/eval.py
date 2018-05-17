import argparse
import numpy as np
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str)
args = parser.parse_args()
log_dir = args.log_dir
if log_dir is None:
    raise ValueError('log_dir not found')
elif not os.path.exists(log_dir):
    raise ValueError('could not find {}'.format(log_dir))

scores_all = np.loadtxt(os.path.join(log_dir, 'scores.txt'))
scores_not_nan = scores_all[~np.isnan(scores_all)]
scores = scores_not_nan[-10:]
print(scores_all)
print('Mean: {:.4f}, Stderr: {:.4f}'.format(np.mean(scores), np.std(scores)))

#scores_wl_all = np.loadtxt(os.path.join(log_dir, 'scores_without_label.txt'))
#scores_wl_not_nan = scores_wl_all[~np.isnan(scores_wl_all)]
#scores_wl = scores_wl_not_nan[-10:]
#print('Without label. Mean: {:.4f}, Stderr: {:.4f}'.format(
#    np.mean(scores_wl), np.std(scores_wl)))
pdb.set_trace()
