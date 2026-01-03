"""
数据蒸馏使用示例，测试使用
"""
import torch
from .dataset_loader import DatasetLoader
from .distillation_manager import DistillationManager
from .config_template import create_config


def main():
    # 1. 选择数据集和蒸馏方法
    dataset_name = 'cifar10'  # 可选: cifar10, cifar100, mnist, fashionmnist, custom
    method = 'gradient_matching'  # 可选: gradient_matching, dc_dsa, hybrid
    ipc = 200  # 每类图像数

    # 2. 创建配置
    config = create_config(dataset_name, method, ipc)

    # 3. 配置数据集参数
    dataset_config = {
        'name': dataset_name,
        'data_dir': './data',
        'batch_size': 256,
        'num_workers': 4,
        'download': True,
    }

    # 对于自定义数据集
    if dataset_name == 'custom':
        dataset_config.update({
            'data_dir': './custom_data',  # 自定义数据路径
            'image_size': 224,  # 图像大小
            'split_ratio': 0.8,  # 训练/测试划分比例
        })

    # 4. 加载数据集
    print(f"加载数据集: {dataset_name}")
    dataset_loader = DatasetLoader(dataset_config)
    train_loader, test_loader = dataset_loader.get_dataloaders()

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    print(f"类别数: {dataset_loader.get_num_classes()}")

    # 5. 创建蒸馏管理器
    device = torch.device(config['device'])
    distill_manager = DistillationManager(config, device)

    # 6. 执行蒸馏（可选传入教师模型）
    teacher_model = None  # 可以加载预训练模型
    print(f"\n开始{method}数据蒸馏...")

    synthetic_data, synthetic_labels = distill_manager.distill(
        dataset_loader, train_loader, teacher_model
    )

    # 7. 创建蒸馏数据集
    class DistilledDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            img = self.data[idx]
            label = self.labels[idx]

            if self.transform:
                img = self.transform(img)

            return img, label

    # 使用与训练相同的转换（无增强）
    transform = dataset_loader.get_transforms(augment=False)
    distilled_dataset = DistilledDataset(synthetic_data, synthetic_labels, transform)

    # 8. 创建数据加载器
    distilled_loader = torch.utils.data.DataLoader(
        distilled_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    print(f"\n蒸馏完成！")
    print(f"得到 {len(synthetic_data)} 个合成样本")
    print(f"每个类别有 {ipc} 个样本")
    print(f"现在可以使用 distilled_loader 训练模型了！")

    # 9. 验证蒸馏数据
    print("\n验证蒸馏数据...")
    sample_batch = next(iter(distilled_loader))
    images, labels = sample_batch
    print(f"批大小: {images.shape}")
    print(f"标签: {labels[:10].tolist()}")  # 显示前10个标签

    return distilled_loader, synthetic_data, synthetic_labels


if __name__ == "__main__":
    distilled_loader, synthetic_data, synthetic_labels = main()