#!/bin/bash


# Wrapper script to run 2-, 4-, and 10-D models, and do eval.
#
# Run by using the following bash commands:
#   bash run_panel.sh 'run'
#   bash run_panel.sh 'eval'
# Note: For run-time performance, comment out the parallel operator '&' so that
#   each run gets the same resources.
# Note: This script calls the sub-task script run_short_panel.sh.


# SET UP list of model names to run and eval.
#declare -a model_names=('ce_iw' 'ce_miw' 'ce_sniw' 'ce_conditional' 'ce_upsample' 'mmd_iw' 'mmd_miw' 'mmd_sniw' 'mmd_upsample')

# NOTE: THESE TWO ARE RUNNING NOW
#declare -a model_names=('ce_iw' 'ce_miw' 'ce_sniw' 'ce_upsample' 'ce_conditional')
#declare -a model_names=('ce_iw' 'ce_conditional')
declare -a model_names=('ce_miw' 'ce_sniw' 'ce_upsample')


# RUN the models.
if [ "$1" == 'run' ]; then
  for model in "${model_names[@]}" ; do
    bash run_short_panel.sh $model &
    echo "Running ${model}."
  done
# EVAL the models.
elif [ "$1" == 'eval' ]; then
  # Run eval script for each model.
  for model in "${model_names[@]}" ; do
    python eval_short_panel.py $model
    echo "Evaluating ${model}."
  done
else
  echo "To run, add either the 'run' or 'eval' argument."
fi
