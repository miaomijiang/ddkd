"""
数据蒸馏配置文件模板，测试使用
"""

import torch


def create_config(dataset_name='cifar10', method='gradient_matching', ipc=2):
    """创建配置字典"""

    # 基础配置
    base_config = {
        'dataset': dataset_name,
        'method': method,
        'ipc': ipc,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # 方法特定配置
    if method == 'gradient_matching':
        method_config = {
            'synthesis_steps': 2,
            'inner_loop': 1,
            'lr_syn': 0.01,
            'lr_net': 0.01,
            'init_method': 'real',
            'use_momentum': True,
            'use_scheduler': True,
            'reg_type': 'l2',
            'reg_weight': 0.001,
        }
    elif method == 'dc_dsa':
        method_config = {
            'synthesis_steps': 1,
            'lr': 0.1,
            'weight_decay': 5e-4,
            'use_teacher': False,
            'augmentation_strength': 0.1,
            'reg_weight': 0.001,
        }
    elif method == 'hybrid':
        method_config = {
            'gm_steps': 1000,
            'dsa_steps': 500,
            'refine_steps': 200,
            'refine_lr': 0.01,
        }
    else:
        method_config = {}

    # 数据集特定配置
    if dataset_name == 'cifar10':
        dataset_config = {
            'image_size': 32,
            'synthesis_steps': 2 if method == 'gradient_matching' else 1,
        }
    elif dataset_name == 'cifar100':
        dataset_config = {
            'image_size': 32,
            'synthesis_steps': 3000 if method == 'gradient_matching' else 1500,
            'ipc': min(ipc, 100),  # CIFAR100类别多，每类样本数可以少一些
        }
    elif dataset_name in ['mnist', 'fashionmnist']:
        dataset_config = {
            'image_size': 28,
            'synthesis_steps': 1000,
        }
    elif dataset_name == 'custom':
        dataset_config = {
            'image_size': 224,
            'synthesis_steps': 5000,
        }
    else:
        dataset_config = {}

    # 合并配置
    config = {**base_config, **method_config, **dataset_config}

    return config