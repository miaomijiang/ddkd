#!/bin/bash

# 创建目录
mkdir -p data
mkdir -p experiments

# 运行CIFAR-10实验
python main.py \
    --config cifar10.yaml \
    --epochs 50 \
    --ipc 10 \
    --teacher resnet18 \
    --student resnet18