#!/usr/bin/env python3
"""
主程序 - 数据蒸馏+知识蒸馏完整流程
"""
import os
import sys
import torch
import argparse
import json
import time
import warnings
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 忽略Matplotlib字体警告
warnings.filterwarnings("ignore", message="Glyph.*missing from font")

from src.utils.config_loader import ConfigLoader
from src.data.dataset_loader import DatasetLoader
from src.data.gradient_matching import GradientMatching
from src.models.model_factory import ModelFactory
from src.models.knowledge_distillation import KnowledgeDistillation
from src.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="数据蒸馏+知识蒸馏")
    parser.add_argument("--config", type=str, default="cifar10.yaml", help="配置文件名称")
    parser.add_argument("--dataset", type=str, help="数据集名称")
    parser.add_argument("--teacher", type=str, help="教师模型名称")
    parser.add_argument("--student", type=str, help="学生模型名称")
    parser.add_argument("--epochs", type=int, help="训练轮数")
    parser.add_argument("--ipc", type=int, help="每类合成图像数量")
    parser.add_argument("--no-distill", action="store_true", help="跳过知识蒸馏")
    parser.add_argument("--no-synthetic", action="store_true", help="跳过合成数据生成")

    args = parser.parse_args()

    print("=" * 80)
    print("数据蒸馏 + 知识蒸馏完整流程")
    print("=" * 80)

    # 1. 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load(args.config)

    # 覆盖配置参数
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.teacher:
        config['teacher']['name'] = args.teacher
    if args.student:
        config['student']['name'] = args.student
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.ipc:
        config['data_distillation']['ipc'] = args.ipc

    # 2. 设置设备
    device = torch.device(config['project']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # CUDA优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 3. 数据导入
    print("\n[1/5] 数据导入...")
    dataset_loader = DatasetLoader(config['dataset'])
    train_loader, test_loader = dataset_loader.get_dataloaders()
    num_classes = dataset_loader.get_num_classes()

    print(f"数据集: {config['dataset']['name']}")
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    print(f"类别数: {num_classes}")

    # 4. 数据蒸馏（梯度匹配）
    synthetic_loader = None
    if not args.no_synthetic and config['data_distillation']['enabled']:
        print("\n[2/5] 数据蒸馏（梯度匹配）...")
        gradient_matcher = GradientMatching(config['data_distillation'], device)
        synthetic_images, synthetic_labels = gradient_matcher.distill(train_loader, num_classes)

        # 创建合成数据加载器
        synthetic_dataset = torch.utils.data.TensorDataset(synthetic_images, synthetic_labels)
        synthetic_loader = torch.utils.data.DataLoader(
            synthetic_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=True,
            num_workers=config['dataset']['num_workers']
        )

        print(f"生成合成数据: {len(synthetic_images)} 张图像")
        print(f"压缩比例: {len(train_loader.dataset) / len(synthetic_images):.1f}x")
    else:
        print("\n[2/5] 跳过数据蒸馏...")

    # 5. 创建模型
    print("\n[3/5] 创建模型...")
    model_factory = ModelFactory()

    # ========== 关键修复：创建模型时直接传递设备参数 ==========
    teacher = model_factory.create_model(
        config['teacher']['name'],
        num_classes=num_classes,
        pretrained=config['teacher']['pretrained'],
        device=device  # 新增设备参数
    )

    student = model_factory.create_model(
        config['student']['name'],
        num_classes=num_classes,
        pretrained=config['student']['pretrained'],
        device=device  # 新增设备参数
    )

    teacher_size = model_factory.get_model_size(teacher)
    student_size = model_factory.get_model_size(student)

    print(f"教师模型: {config['teacher']['name']}")
    print(f"学生模型: {config['student']['name']}")
    print(f"教师模型参数量: {teacher_size['parameters']:,}")
    print(f"学生模型参数量: {student_size['parameters']:,}")

    # 6. 知识蒸馏
    results = {}
    if not args.no_distill:
        print("\n[4/5] 知识蒸馏训练...")
        start_time = time.time()

        # 使用合成数据或原始数据进行蒸馏
        distill_train_loader = synthetic_loader if synthetic_loader else train_loader

        # 训练教师模型（简单训练）
        print("预训练教师模型...")
        teacher_optimizer = torch.optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)

        # 确保教师模型在训练模式
        teacher.train()

        for epoch in range(10):
            for images, labels in train_loader:
                # 确保数据在正确设备
                images = images.to(device)
                labels = labels.to(device)

                teacher_optimizer.zero_grad()
                outputs = teacher(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                teacher_optimizer.step()
        print("教师模型训练完成")

        # 知识蒸馏
        distill_trainer = KnowledgeDistillation(config['knowledge_distillation'], device)
        distill_history = distill_trainer.train(
            teacher, student, distill_train_loader, test_loader,
            epochs=config['training']['epochs']
        )

        training_time = time.time() - start_time

        # 7. 评估
        print("\n[5/5] 评估模型...")
        evaluator = ModelEvaluator(device)

        # 加载最佳学生模型并确保在正确设备
        student.load_state_dict(torch.load("best_student.pth"))
        student = student.to(device)  # 关键修复：确保加载的模型在正确设备

        # 评估蒸馏后模型
        distill_results = evaluator.evaluate(student, test_loader)

        # 评估基准模型（单独训练学生模型）
        print("评估基准学生模型...")
        # ========== 关键修复：创建基准模型时传递设备参数 ==========
        baseline_student = model_factory.create_model(
            config['student']['name'],
            num_classes=num_classes,
            pretrained=config['student']['pretrained'],
            device=device  # 新增设备参数
        )

        # 简单训练基准模型
        baseline_optimizer = torch.optim.SGD(baseline_student.parameters(), lr=0.01)

        # 确保基准模型在训练模式
        baseline_student.train()

        for epoch in range(config['training']['epochs'] // 2):
            for images, labels in distill_train_loader:
                # 确保数据在正确设备
                images = images.to(device)
                labels = labels.to(device)

                baseline_optimizer.zero_grad()
                outputs = baseline_student(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()
                baseline_optimizer.step()

        # 评估基准模型
        baseline_student.eval()
        baseline_results = evaluator.evaluate(baseline_student, test_loader)

        # 比较结果
        comparison = evaluator.compare_models(baseline_student, student, test_loader)

        results = {
            'config': config,
            'baseline': baseline_results,
            'distilled': distill_results,
            'comparison': comparison['improvement'],
            'training_time': training_time,
            'distill_history': distill_history
        }

        # 打印结果
        print("\n" + "=" * 80)
        print("实验结果")
        print("=" * 80)
        print(f"训练时间: {training_time:.1f} 秒")
        print(f"基准模型 Top-1准确率: {baseline_results['accuracy_top1']:.2f}%")
        print(f"蒸馏模型 Top-1准确率: {distill_results['accuracy_top1']:.2f}%")
        print(f"准确率提升: {comparison['improvement']['accuracy_top1']:+.2f}%")
        print(f"推理时间: {distill_results['inference_time_per_sample'] * 1000:.2f} ms")
        print(f"模型大小: {distill_results['model_size_mb']:.1f} MB")

    # 8. 保存结果
    if results:
        experiment_dir = Path("experiments") / config['project']['experiment_name']
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(experiment_dir / "config.yaml", 'w') as f:
            import yaml
            yaml.dump(config, f, default_flow_style=False)

        # 保存结果
        with open(experiment_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=4, default=str)

        print(f"\n结果已保存到: {experiment_dir}")

    print("\n" + "=" * 80)
    print("流程完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()