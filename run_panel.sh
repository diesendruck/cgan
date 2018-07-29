#!/bin/bash


# Wrapper script to run 2-, 4-, and 10-D models, and do eval.
#
# Run by using the following bash commands:
#   bash run_panel.sh 'run'
#   bash run_panel.sh 'eval'


# SET UP list of model names to run and eval.
declare -a model_names=('ce_iw' 'ce_sn' 'ce_miw' 'mmd_iw' 'mmd_sn' 'mmd_miw' 'cgan' 'upsample')
#declare -a model_names=('ce_iw' 'ce_sn')
#declare -a model_names=('cgan' 'upsample')

# RUN the models.
if [ "$1" == 'run' ]; then
  for model in "${model_names[@]}" ; do
    bash run_short_panel.sh $model &
  done
fi

# EVAL the models.
if [ "$1" == 'eval' ]; then
  # Run eval script for each model.
  for model in "${model_names[@]}" ; do
    python eval_short_panel.py $model
  done
fi
