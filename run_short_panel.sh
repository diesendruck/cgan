#!/bin/bash


NUM_RUNS=20

################################################################################
# CE GAN
if [ "$1" == 'ce_iw' ]; then
  # IWGAN with importance weights.
  for i in $( seq 1 $NUM_RUNS ); do
    rm -rf results/ce_iw_dim2_run${i}; python iwgan.py --tag='iw_dim2_run'${i} --data_dim=2 --estimator='iw' &
    rm -rf results/ce_iw_dim4_run${i}; python iwgan.py --tag='iw_dim4_run'${i} --data_dim=4 --estimator='iw' &
    rm -rf results/ce_iw_dim10_run${i}; python iwgan.py --tag='iw_dim10_run'${i} --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'ce_sn' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # IWGAN with self-normalized importance weights.
    rm -rf results/ce_sn_dim2_run${i}; python iwgan.py --tag='sn_dim2_run'${i} --data_dim=2 --estimator='sn' &
    rm -rf results/ce_sn_dim4_run${i}; python iwgan.py --tag='sn_dim4_run'${i} --data_dim=4 --estimator='sn' &
    rm -rf results/ce_sn_dim10_run${i}; python iwgan.py --tag='sn_dim10_run'${i} --data_dim=10 --estimator='sn' &
  done
fi

if [ "$1" == 'ce_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # IWGAN with median importance weights.
    rm -rf results/ce_miw_dim2_run${i}; python iwgan_mom.py --tag='miw_dim2_run'${i} --data_dim=2 &
    rm -rf results/ce_miw_dim4_run${i}; python iwgan_mom.py --tag='miw_dim4_run'${i} --data_dim=4 &
    rm -rf results/ce_miw_dim10_run${i}; python iwgan_mom.py --tag='miw_dim10_run'${i} --data_dim=10 &
  done
fi

################################################################################
# MMD GAN
if [ "$1" == 'mmd_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with importance weights.
    rm -rf results/mmd_iw_dim2_run${i}; python mmdgan.py --tag='iw_dim2_run'${i} --data_dim=2 --estimator='iw' &
    rm -rf results/mmd_iw_dim4_run${i}; python mmdgan.py --tag='iw_dim4_run'${i} --data_dim=4 --estimator='iw' &
    rm -rf results/mmd_iw_dim10_run${i}; python mmdgan.py --tag='iw_dim10_run'${i} --data_dim=10 --estimator='iw' &
  done
fi

if [ "$1" == 'mmd_sn' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with self-normalized importance weights.
    rm -rf results/mmd_sn_dim2_run${i}; python mmdgan.py --tag='sn_dim2_run'${i} --data_dim=2 --estimator='sn' &
    rm -rf results/mmd_sn_dim4_run${i}; python mmdgan.py --tag='sn_dim4_run'${i} --data_dim=4 --estimator='sn' &
    rm -rf results/mmd_sn_dim10_run${i}; python mmdgan.py --tag='sn_dim10_run'${i} --data_dim=10 --estimator='sn' &
  done
fi

if [ "$1" == 'mmd_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # MMDGAN with median of means on importance weights.
    rm -rf results/mmd_miw_dim2_run${i}; python mmdgan_mom.py --tag='miw_dim2_run'${i} --data_dim=2 &
    rm -rf results/mmd_miw_dim4_run${i}; python mmdgan_mom.py --tag='miw_dim4_run'${i} --data_dim=4 &
    rm -rf results/mmd_miw_dim10_run${i}; python mmdgan_mom.py --tag='miw_dim10_run'${i} --data_dim=10 &
  done
fi


################################################################################
# CGAN
if [ "$1" == 'cgan' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # CGAN
    rm -rf results/cgan_dim2_run${i}; python cgan.py --tag='dim2_run'${i} --data_dim=2 &
    rm -rf results/cgan_dim4_run${i}; python cgan.py --tag='dim4_run'${i} --data_dim=4 &
    rm -rf results/cgan_dim10_run${i}; python cgan.py --tag='dim10_run'${i} --data_dim=10 &
  done
fi

################################################################################
# UPSAMPLE
if [ "$1" == 'upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    # UPSAMPLE
    rm -rf results/upsample_dim2_run${i}; python upsample.py --tag='dim2_run'${i} --data_dim=2 &
    rm -rf results/upsample_dim4_run${i}; python upsample.py --tag='dim4_run'${i} --data_dim=4 &
    rm -rf results/upsample_dim10_run${i}; python upsample.py --tag='dim10_run'${i} --data_dim=10 &
  done
fi
