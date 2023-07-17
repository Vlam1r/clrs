#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

var=$1

seed=$((42 + var % 5))
var=$((var/5))

soft=$((var%2))
var=$((var/2))

dec=$((var%2))
var=$((var/2))

dir=$RANDOM

echo $dir

if [ "$soft" = 1 ]; then
  if [ "$dec" = 1 ]; then
    python -m clrs.examples.run --dataset_path ~/rds --checkpoint_path ~/rds/clrs/$dir \
           --processor_type triplet_gmpnn --train_steps $3 \
           --seed $seed --algorithms $2 \
           --noise_injection_strategy Noisefree --softmax_reduction True --decay 0.9
  else
    python -m clrs.examples.run --dataset_path ~/rds --checkpoint_path ~/rds/clrs/$dir \
           --processor_type triplet_gmpnn --train_steps $3 \
           --seed $seed --algorithms $2 \
           --noise_injection_strategy Noisefree --softmax_reduction True --decay 1.0
  fi
else
  if [ "$dec" = 1 ]; then
    python -m clrs.examples.run --dataset_path ~/rds --checkpoint_path ~/rds/clrs/$dir \
           --processor_type triplet_gmpnn --train_steps $3 \
           --seed $seed --algorithms $2 \
           --noise_injection_strategy Noisefree --decay 0.9
  else
    python -m clrs.examples.run --dataset_path ~/rds --checkpoint_path ~/rds/clrs/$dir \
           --processor_type triplet_gmpnn --train_steps $3 \
           --seed $seed --algorithms $2 \
           --noise_injection_strategy Noisefree --decay 1.0
  fi
fi
