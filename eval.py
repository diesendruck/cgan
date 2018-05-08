import numpy as np
import os
import pdb

log_dir = 'iwgan_out_bar'
log_dir = 'cgan_out_bar'

scores_all = np.loadtxt(os.path.join(log_dir, 'scores.txt'))
scores = scores_all[-20:]
print('Mean: {:.4f}, Stderr: {:.4f}'.format(np.mean(scores), np.std(scores)))
pdb.set_trace()
