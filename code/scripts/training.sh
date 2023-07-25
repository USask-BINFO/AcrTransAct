#!/bin/bash

COMMON_ARGS="python3 code/training.py
            --excl_mode=0
            --monitor_ckpt=val_f1_score
            --optimize_metric=F1_CV
            --esm_version=8m
            --wandb_log=True
            --cross_val=True
            --run_mode=train
            --do_sweep=False"

for model_type in CNN LSTM
do
    for feature_mode in 1 2 3 4 
    do
        ARGS="--model_type=$model_type
              --feature_mode=$feature_mode
              "
        $COMMON_ARGS $ARGS
    done
done

# train on bigger esm models
COMMON_ARGS="python3 code/training.py
            --excl_mode=0
            --monitor_ckpt=val_f1_score
            --optimize_metric=F1_CV
            --do_sweep=False
            --wandb_log=True
            --cross_val=True
            --run_mode=train
            --feature_mode=1"

for model_type in CNN LSTM
do
    for esm_version in 35m 150m 
    do
        ARGS="--model_type=$model_type
              --esm_version=$esm_version
              "
        $COMMON_ARGS $ARGS
    done
done
