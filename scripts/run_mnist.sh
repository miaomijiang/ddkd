#!/bin/bash

python main.py \
    --config mnist.yaml \
    --epochs 30 \
    --ipc 5 \
    --teacher simple_cnn \
    --student simple_cnn