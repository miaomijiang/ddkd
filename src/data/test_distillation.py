"""
测试数据蒸馏
"""
import torch
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from .dataset_loader import DatasetLoader
from .distillation_manager import DistillationManager


def test_gradient_matching():
    """测试梯度匹配方法"""
    print("=== 测试梯度匹配 ===")

    # 配置
    config = {
        'method': 'gradient_matching',
        'ipc': 50,  # 每类50张
        'synthesis_steps': 500,  # 测试用，步数减少
        'lr_syn': 0.01,
        'lr_net': 0.01,
        'inner_loop': 1,
        'init_method': 'real',
    }

    # 数据集配置
    dataset_config = {
        'name': 'cifar10',
        'data_dir': './data',
        'batch_size': 128,
        'num_workers': 2,
        'download': True,
    }

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据集
    dataset_loader = DatasetLoader(dataset_config)
    train_loader, test_loader = dataset_loader.get_dataloaders()

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"类别数: {dataset_loader.get_num_classes()}")

    # 创建蒸馏管理器
    manager = DistillationManager(config, device)

    # 执行蒸馏
    synthetic_data, synthetic_labels = manager.distill(
        dataset_loader, train_loader
    )

    print(f"\n蒸馏完成!")
    print(f"合成数据形状: {synthetic_data.shape}")
    print(f"合成标签形状: {synthetic_labels.shape}")

    # 保存结果
    torch.save({
        'data': synthetic_data,
        'labels': synthetic_labels
    }, 'gradient_matching_result.pt')

    print("结果已保存到 gradient_matching_result.pt")

    return synthetic_data, synthetic_labels


def test_dc_dsa():
    """测试DC-DSA方法"""
    print("\n=== 测试DC-DSA ===")

    config = {
        'method': 'dc_dsa',
        'ipc': 50,
        'synthesis_steps': 300,  # 测试用，步数减少
        'lr': 0.1,
    }

    dataset_config = {
        'name': 'cifar10',
        'data_dir': './data',
        'batch_size': 128,
        'num_workers': 2,
        'download': True,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_loader = DatasetLoader(dataset_config)
    train_loader, _ = dataset_loader.get_dataloaders()

    manager = DistillationManager(config, device)
    synthetic_data, synthetic_labels = manager.distill(dataset_loader, train_loader)

    print(f"DC-DSA完成!")
    torch.save({
        'data': synthetic_data,
        'labels': synthetic_labels
    }, 'dc_dsa_result.pt')

    return synthetic_data, synthetic_labels


def main():
    """主函数"""
    print("数据蒸馏测试开始...")

    # 测试梯度匹配
    gm_data, gm_labels = test_gradient_matching()

    # 测试DC-DSA（可选）
    # dsa_data, dsa_labels = test_dc_dsa()

    print("\n所有测试完成!")


if __name__ == "__main__":
    main()