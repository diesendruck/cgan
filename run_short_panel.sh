#!/bin/bash


# This script defines specific models to run. It is called by the main script
# called run_panel.sh.
#
# NOTE: For run-time performance measurements, comment out the parallel
#   operator ' #&' so that each run gets the same resources.


NUM_RUNS=10
MAX_STEP=2001
LOG_STEP=500
BATCH_SIZE=64
LEARNING_RATE=1e-3
LATENT_DIM=10 # For generate_data(), not generate_data2(). Dim of latent features.
NOISE_DIM=10
H_DIM=10
DO_DIM2=true
DO_DIM4=true
DO_DIM10=true


################################################################################
# CE GAN - BASELINE.
if [ "$1" == 'ce_baseline' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_baseline_dim2_run${i}; python gan_ce_iw.py \
        --tag='baseline_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='baseline' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_baseline_dim4_run${i}; python gan_ce_iw.py \
        --tag='baseline_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='baseline' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_baseline_dim10_run${i}; python gan_ce_iw.py \
        --tag='baseline_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='baseline' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
  done
fi

################################################################################
# CE GAN
if [ "$1" == 'ce_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_iw_dim2_run${i}; python gan_ce_iw.py \
        --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='iw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_iw_dim4_run${i}; python gan_ce_iw.py \
        --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='iw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_iw_dim10_run${i}; python gan_ce_iw.py \
        --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='iw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
  done
fi

if [ "$1" == 'ce_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_miw_dim2_run${i}; python gan_ce_miw.py \
        --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_miw_dim4_run${i}; python gan_ce_miw.py \
        --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_miw_dim10_run${i}; python gan_ce_miw.py \
        --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
  done
fi

if [ "$1" == 'ce_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_sniw_dim2_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_sniw_dim4_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_sniw_dim10_run${i}; python gan_ce_iw.py \
        --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
  done
fi

# CONDITIONAL
if [ "$1" == 'ce_conditional' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_conditional_dim2_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim2_run'${i} --max_step=$MAX_STEP --data_dim=2 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_conditional_dim4_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim4_run'${i} --max_step=$MAX_STEP --data_dim=4 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_conditional_dim10_run${i}; python gan_ce_conditional.py \
        --tag='conditional_dim10_run'${i} --max_step=$MAX_STEP --data_dim=10 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
  done
fi

# UPSAMPLE
if [ "$1" == 'ce_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/ce_upsample_dim2_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP --data_dim=2\
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random' --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/ce_upsample_dim4_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP --data_dim=2 \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random' --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/ce_upsample_dim10_run${i}; python gan_ce_upsample.py \
        --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP --data_dim=10\
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --sampling='random' --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
  done
fi


################################################################################
# MMD GAN
if [ "$1" == 'mmd_iw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_iw_dim2_run${i}; python gan_mmd_iw.py --data_dim=2 \
        --tag='iw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw' \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_iw_dim4_run${i}; python gan_mmd_iw.py --data_dim=4 \
        --tag='iw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw' \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_iw_dim10_run${i}; python gan_mmd_iw.py --data_dim=10 \
        --tag='iw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --estimator='iw' \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
  done
fi

if [ "$1" == 'mmd_miw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_miw_dim2_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_miw_dim4_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_miw_dim10_run${i}; python gan_mmd_miw.py \
        --tag='miw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
  done
fi

if [ "$1" == 'mmd_sniw' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_sniw_dim2_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim2_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=2 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_sniw_dim4_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim4_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=4 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_sniw_dim10_run${i}; python gan_mmd_iw.py \
        --tag='sniw_dim10_run'${i} --max_step=$MAX_STEP --batch_size=$BATCH_SIZE \
        --learning_rate=$LEARNING_RATE --log_step=$LOG_STEP --data_dim=10 \
        --estimator='sniw' --latent_dim=$LATENT_DIM --noise_dim=$NOISE_DIM \
        --h_dim=$H_DIM  &
    fi
  done
fi

# UPSAMPLE
if [ "$1" == 'mmd_upsample' ]; then
  for i in $( seq 1 $NUM_RUNS ); do
    if [ "$DO_DIM2" = true ] ; then
      rm -rf results/mmd_upsample_dim2_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim2_run'${i} --max_step=$MAX_STEP \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --data_dim=2 --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM4" = true ] ; then
      rm -rf results/mmd_upsample_dim4_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim4_run'${i} --max_step=$MAX_STEP \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --data_dim=4 --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
    if [ "$DO_DIM10" = true ] ; then
      rm -rf results/mmd_upsample_dim10_run${i}; python gan_mmd_upsample.py \
        --tag='upsample_dim10_run'${i} --max_step=$MAX_STEP \
        --batch_size=$BATCH_SIZE --learning_rate=$LEARNING_RATE \
        --log_step=$LOG_STEP --data_dim=10 --latent_dim=$LATENT_DIM \
        --noise_dim=$NOISE_DIM --h_dim=$H_DIM  &
    fi
  done
fi
