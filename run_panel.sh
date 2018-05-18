#!/bin/bash


# IWGAN with importance weights.
rm -rf results/iwgan_2iw; python iwgan.py --tag='2iw' --data_dim=2 --estimator='iw' &
rm -rf results/iwgan_4iw; python iwgan.py --tag='4iw' --data_dim=4 --estimator='iw' &
rm -rf results/iwgan_10iw; python iwgan.py --tag='10iw' --data_dim=10 --estimator='iw' &

# IWGAN with self-normalized importance weights.
rm -rf results/iwgan_2sn; python iwgan.py --tag='2sn' --data_dim=2 --estimator='sn' &
rm -rf results/iwgan_4sn; python iwgan.py --tag='4sn' --data_dim=4 --estimator='sn' &
rm -rf results/iwgan_10sn; python iwgan.py --tag='10sn' --data_dim=10 --estimator='sn' &

# IWGAN with median importance weights.
rm -rf results/iwgan_2mom; python iwgan_mom.py --tag='2mom' --data_dim=2 &
rm -rf results/iwgan_4mom; python iwgan_mom.py --tag='4mom' --data_dim=4 &
rm -rf results/iwgan_10mom; python iwgan_mom.py --tag='10mom' --data_dim=10 &


# MMDGAN with importance weights.
rm -rf results/mmdgan_2iw; python mmdgan.py --tag='2iw' --data_dim=2 --estimator='iw' &
rm -rf results/mmdgan_4iw; python mmdgan.py --tag='4iw' --data_dim=4 --estimator='iw' &
rm -rf results/mmdgan_10iw; python mmdgan.py --tag='10iw' --data_dim=10 --estimator='iw' &

# MMDGAN with self-normalized importance weights.
rm -rf results/mmdgan_2sn; python mmdgan.py --tag='2sn' --data_dim=2 --estimator='sn' &
rm -rf results/mmdgan_4sn; python mmdgan.py --tag='4sn' --data_dim=4 --estimator='sn' &
rm -rf results/mmdgan_10sn; python mmdgan.py --tag='10sn' --data_dim=10 --estimator='sn' &

# MMDGAN with median of means on importance weights.
rm -rf results/mmdgan_2mom; python mmdgan_mom.py --tag='2mom' --data_dim=2 &
rm -rf results/mmdgan_4mom; python mmdgan_mom.py --tag='4mom' --data_dim=4 &
rm -rf results/mmdgan_10mom; python mmdgan_mom.py --tag='10mom' --data_dim=10 &


# CGAN
rm -rf results/cgan_2d; python cgan.py --tag='2d' --data_dim=2
rm -rf results/cgan_4d; python cgan.py --tag='4d' --data_dim=4
rm -rf results/cgan_10d; python cgan.py --tag='10d' --data_dim=10

# UPSAMPLE
rm -rf results/upsample_2d; python upsample.py --tag='2d' --data_dim=2
rm -rf results/upsample_4d; python upsample.py --tag='4d' --data_dim=4
rm -rf results/upsample_10d; python upsample.py --tag='10d' --data_dim=10
