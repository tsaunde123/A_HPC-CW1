#!/bin/bash
module add libs/tensorflow/1.2
salloc -N 1 --partition gpu_veryshort -t 00:10:00 --gres=gpu:1
