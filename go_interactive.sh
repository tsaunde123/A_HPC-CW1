#!/bin/bash
module add libs/tensorflow/1.2
srun -p gpu --gres=gpu:1  -t 0-01:00 --mem=8G --pty bash
