#!/usr/bin/env python3
"""
简化版训练脚本
"""
import argparse
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cifar10.yaml")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--teacher", type=str)
    parser.add_argument("--student", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--ipc", type=int, default=10)

    args = parser.parse_args()

    main()