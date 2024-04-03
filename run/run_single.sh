#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
python run/main.py --cfg run/configs/example_test.yaml --repeat 1 --auto_select_device

# Ensemble over multiple seeds
#python run/eval_ensemble.py --repeat 3 --cfg results/f1-position_optim.base_lr_0.01_0.001_model.channels_128_256_model.feature_dropout_0.0_0.2_0.4_model.mask_features_False_True/f1-position_optim.base_lr_0.001_model.channels_256_model.feature_dropout_0.0_model.mask_features_False_run/config.yaml --auto_select_device