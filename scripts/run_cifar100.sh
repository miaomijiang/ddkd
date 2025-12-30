#!/bin/bash

python main.py \
    --config cifar100.yaml \
    --epochs 100 \
    --ipc 5 \
    --teacher resnet34 \
    --student resnet18