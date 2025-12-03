import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import os
import numpy as np
import time
from torchvision import models
from utils.data_loader import get_data_loaders
from utils.loss_functions import DistillLoss, gradient_matching_loss
from models import get_model
from utils.logger import TensorboardLogger
# 导入新配置（删除旧的DATASET_TYPE相关导入）
from config.config import (
    DEVICE, DATASET_NAME, DATASET_META, DATA_SOURCE, SYNTHETIC_DATA_PATH,
    RUN_DATA_DISTILL, IPC, DATA_DISTILL_EPOCHS, DATA_DISTILL_LR,
    TEACHER_MODEL_TYPE, STUDENT_MODEL_TYPE,
    BATCH_SIZE, LEARNING_RATE, EPOCHS, MOMENTUM, WEIGHT_DECAY,
    LR_SCHEDULER, STEP_SIZE, GAMMA, TEMPERATURE, DISTILL_WEIGHT,
    GRADIENT_MATCH_WEIGHT, SAVE_DIR, SAVE_FREQ, LOG_DIR, SEED
)

# 固定随机种子（保证复现性）
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# （train_one_epoch、validate函数保持不变，无需修改）

def main():
    # 1. 初始化组件
    print("=" * 60)
    print("开始两阶段蒸馏：数据蒸馏 → 知识蒸馏")
    print(f"配置信息：数据集={DATASET_NAME} | 训练集来源={DATA_SOURCE} | 执行数据蒸馏={RUN_DATA_DISTILL}")
    print("=" * 60)

    # 2. 第一阶段：数据蒸馏（生成合成数据集）
    if RUN_DATA_DISTILL:
        print("\n🚀 启动第一阶段：数据蒸馏（生成合成数据集）")
        train_loader_distill, _, _, num_classes = get_data_loaders(stage="DISTILL")
        # 执行数据蒸馏
        from utils.loss_functions import distill_dataset
        distill_dataset(train_loader_distill, num_classes, DEVICE)
    else:
        print("\n⚠️  跳过数据蒸馏阶段（RUN_DATA_DISTILL=False）")
        # 验证合成数据集是否存在（若选择合成数据训练）
        if DATA_SOURCE == "SYNTHETIC" and not os.path.exists(SYNTHETIC_DATA_PATH):
            raise FileNotFoundError(
                f"错误：训练集来源=SYNTHETIC，但合成数据集不存在！\n"
                f"解决方案1：设置RUN_DATA_DISTILL=True生成合成数据\n"
                f"解决方案2：修改DATA_SOURCE=ORIGINAL使用原始数据训练"
            )

    # 3. 第二阶段：知识蒸馏（训练学生模型）
    print("\n🚀 启动第二阶段：知识蒸馏（训练学生模型）")
    train_loader_kd, val_loader, test_loader, num_classes = get_data_loaders(stage="KD")

    # 4. 加载教师/学生模型
    print(f"\n📦 加载模型：教师={TEACHER_MODEL_TYPE} | 学生={STUDENT_MODEL_TYPE} | 类别数={num_classes}")
    teacher_model = get_model(TEACHER_MODEL_TYPE, num_classes).to(DEVICE)
    student_model = get_model(STUDENT_MODEL_TYPE, num_classes).to(DEVICE)

    # 5. 损失函数、优化器、调度器（保持不变）
    distill_criterion = DistillLoss()
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        student_model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    if LR_SCHEDULER == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        print(f"⏱️  学习率调度器：CosineAnnealingLR（T_max={EPOCHS}）")
    else:
        scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
        print(f"⏱️  学习率调度器：StepLR（step_size={STEP_SIZE}, gamma={GAMMA}）")

    # 6. 日志记录器
    logger = TensorboardLogger(log_dir=LOG_DIR)
    print(f"📝 Tensorboard日志目录：{LOG_DIR}")
    print(f"💾 模型保存目录：{SAVE_DIR}")
    print("=" * 60)

    # 7. 训练循环（保持不变）
    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"\n{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{EPOCHS}], LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'=' * 60}")

        train_loss, train_acc = train_one_epoch(
            train_loader_kd, teacher_model, student_model, distill_criterion,
            optimizer, epoch, logger, num_classes
        )

        val_loss, val_acc = validate(val_loader, student_model, val_criterion)
        logger.log_val(val_loss, val_acc, epoch + 1)

        scheduler.step()

        # 保存模型（文件名包含数据集名称，避免冲突）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(SAVE_DIR, f"best_student_{DATASET_NAME}_{DATA_SOURCE.lower()}.pth")
            torch.save(student_model.state_dict(), best_model_path)
            print(f"✅ 保存最佳模型（Val Acc: {best_val_acc:.2f}%）→ {best_model_path}")

        if (epoch + 1) % SAVE_FREQ == 0:
            epoch_model_path = os.path.join(SAVE_DIR,
                                            f"student_epoch_{epoch + 1}_{DATASET_NAME}_{DATA_SOURCE.lower()}.pth")
            torch.save(student_model.state_dict(), epoch_model_path)
            print(f"📌 定期保存模型（Epoch {epoch + 1}）→ {epoch_model_path}")

    # 8. 训练结束
    print("\n" + "=" * 60)
    print(f"两阶段蒸馏完成！🎉")
    print(f"配置汇总：数据集={DATASET_NAME} | 训练集来源={DATA_SOURCE} | IPC={IPC if DATA_SOURCE == 'SYNTHETIC' else 'N/A'}")
    print(f"最佳验证准确率：{best_val_acc:.2f}%")
    print(f"最佳模型路径：{best_model_path}")
    print("=" * 60)
    logger.close()


if __name__ == "__main__":
    main()