#!/bin/bash


NUM_RUNS=2

################################################################################
# CE GAN
if [ "$1" == 'ce_iw' ]; then
  # IWGAN with importance weights.
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/iwgan_iw2_${i}; python iwgan.py --tag='iw2_'${i} --data_dim=2 --estimator='iw' &
    rm -rf results/iwgan_iw4_${i}; python iwgan.py --tag='iw4_'${i} --data_dim=4 --estimator='iw' &
    rm -rf results/iwgan_iw10_${i}; python iwgan.py --tag='iw10_'${i} --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'ce_sn' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # IWGAN with self-normalized importance weights.
    rm -rf results/iwgan_sn2_${i}; python iwgan.py --tag='sn2_'${i} --data_dim=2 --estimator='sn' &
    rm -rf results/iwgan_sn4_${i}; python iwgan.py --tag='sn4_'${i} --data_dim=4 --estimator='sn' &
    rm -rf results/iwgan_sn10_${i}; python iwgan.py --tag='sn10_'${i} --data_dim=10 --estimator='sn' &
  done
fi

if [ "$1" == 'ce_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # IWGAN with median importance weights.
    rm -rf results/iwgan_miw2_${i}; python iwgan_mom.py --tag='miw2_'${i} --data_dim=2 &
    rm -rf results/iwgan_miw4_${i}; python iwgan_mom.py --tag='miw4_'${i} --data_dim=4 &
    rm -rf results/iwgan_miw10_${i}; python iwgan_mom.py --tag='miw10_'${i} --data_dim=10 &
  done
fi

################################################################################
# MMD GAN
if [ "$1" == 'mmd_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with importance weights.
    rm -rf results/mmdgan_iw2_${i}; python mmdgan.py --tag='iw2_'${i} --data_dim=2 --estimator='iw' &
    rm -rf results/mmdgan_iw4_${i}; python mmdgan.py --tag='iw4_'${i} --data_dim=4 --estimator='iw' &
    rm -rf results/mmdgan_iw10_${i}; python mmdgan.py --tag='iw10_'${i} --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'mmd_sn' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with self-normalized importance weights.
    rm -rf results/mmdgan_sn2_${i}; python mmdgan.py --tag='sn2_'${i} --data_dim=2 --estimator='sn' &
    rm -rf results/mmdgan_sn4_${i}; python mmdgan.py --tag='sn4_'${i} --data_dim=4 --estimator='sn' &
    rm -rf results/mmdgan_sn10_${i}; python mmdgan.py --tag='sn10_'${i} --data_dim=10 --estimator='sn' &
  done
fi

if [ "$1" == 'mmd_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with median of means on importance weights.
    rm -rf results/mmdgan_miw2_${i}; python mmdgan_mom.py --tag='miw2_'${i} --data_dim=2 &
    rm -rf results/mmdgan_miw4_${i}; python mmdgan_mom.py --tag='miw4_'${i} --data_dim=4 &
    rm -rf results/mmdgan_miw10_${i}; python mmdgan_mom.py --tag='miw10_'${i} --data_dim=10 &
  done
fi


################################################################################
# CGAN
if [ "$1" == 'cgan' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # CGAN
    rm -rf results/cgan_2d_${i}; python cgan.py --tag='2d_'${i} --data_dim=2 &
    rm -rf results/cgan_4d_${i}; python cgan.py --tag='4d_'${i} --data_dim=4 &
    rm -rf results/cgan_10d_${i}; python cgan.py --tag='10d_'${i} --data_dim=10 &
  done
fi

################################################################################
# UPSAMPLE
if [ "$1" == 'upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # UPSAMPLE
    rm -rf results/upsample_2d_${i}; python upsample.py --tag='2d_'${i} --data_dim=2 &
    rm -rf results/upsample_4d_${i}; python upsample.py --tag='4d_'${i} --data_dim=4 &
    rm -rf results/upsample_10d_${i}; python upsample.py --tag='10d_'${i} --data_dim=10 &
  done
fi
