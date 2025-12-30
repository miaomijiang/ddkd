"""
知识蒸馏模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, Optional


class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失"""
        # 软目标损失
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 硬目标损失
        hard_loss = self.ce_loss(student_logits, labels)

        # 总损失
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return {
            'total': total_loss,
            'soft': soft_loss,
            'hard': hard_loss
        }


class KnowledgeDistillation:
    """知识蒸馏训练器"""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.distill_loss = DistillationLoss(
            temperature=config['temperature'],
            alpha=config['alpha']
        )

    def train(self, teacher: nn.Module, student: nn.Module,
              train_loader: torch.utils.data.DataLoader,
              test_loader: torch.utils.data.DataLoader,
              epochs: int = 50) -> Dict:
        """执行知识蒸馏训练"""
        teacher = teacher.to(self.device)
        student = student.to(self.device)
        teacher.eval()

        # 创建优化器
        optimizer = self._create_optimizer(student)
        scheduler = self._create_scheduler(optimizer, epochs)

        # 训练历史
        history = {
            'train_loss': [], 'train_acc_top1': [], 'train_acc_top5': [],
            'val_loss': [], 'val_acc_top1': [], 'val_acc_top5': [],
            'epoch_time': [], 'learning_rate': []
        }

        best_acc = 0.0

        for epoch in range(epochs):
            start_time = time.time()

            # 训练
            train_metrics = self._train_epoch(
                teacher, student, train_loader, optimizer
            )

            # 验证
            val_metrics = self._validate(student, test_loader)

            # 更新学习率
            scheduler.step()

            # 记录
            epoch_time = time.time() - start_time

            history['train_loss'].append(train_metrics['loss'])
            history['train_acc_top1'].append(train_metrics['acc_top1'])
            history['train_acc_top5'].append(train_metrics['acc_top5'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc_top1'].append(val_metrics['acc_top1'])
            history['val_acc_top5'].append(val_metrics['acc_top5'])
            history['epoch_time'].append(epoch_time)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            # 输出进度
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['acc_top1']:.2f}%, "
                  f"Val Acc: {val_metrics['acc_top1']:.2f}%, "
                  f"Time: {epoch_time:.1f}s")

            # 保存最佳模型
            if val_metrics['acc_top1'] > best_acc:
                best_acc = val_metrics['acc_top1']
                torch.save(student.state_dict(), "best_student.pth")

        history['best_val_acc'] = best_acc
        return history

    def _train_epoch(self, teacher: nn.Module, student: nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     optimizer: torch.optim.Optimizer) -> Dict:
        """训练一个epoch"""
        student.train()

        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 教师预测
            with torch.no_grad():
                teacher_logits = teacher(images)

            # 学生预测
            student_logits = student(images)

            # 计算损失
            losses = self.distill_loss(student_logits, teacher_logits, labels)
            loss = losses['total']

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            total_samples += labels.size(0)

            # Top-1准确率
            _, predicted = student_logits.max(1)
            total_correct_top1 += predicted.eq(labels).sum().item()

            # Top-5准确率
            _, top5_pred = student_logits.topk(5, 1, True, True)
            total_correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc_top1 = 100. * total_correct_top1 / total_samples
        acc_top5 = 100. * total_correct_top5 / total_samples

        return {
            'loss': avg_loss,
            'acc_top1': acc_top1,
            'acc_top5': acc_top5
        }

    def _validate(self, model: nn.Module,
                  test_loader: torch.utils.data.DataLoader) -> Dict:
        """验证模型"""
        model.eval()

        total_loss = 0.0
        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)

                total_loss += loss.item()
                total_samples += labels.size(0)

                # Top-1准确率
                _, predicted = outputs.max(1)
                total_correct_top1 += predicted.eq(labels).sum().item()

                # Top-5准确率
                _, top5_pred = outputs.topk(5, 1, True, True)
                total_correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()

        avg_loss = total_loss / len(test_loader)
        acc_top1 = 100. * total_correct_top1 / total_samples
        acc_top5 = 100. * total_correct_top5 / total_samples

        return {
            'loss': avg_loss,
            'acc_top1': acc_top1,
            'acc_top5': acc_top5
        }

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """创建优化器"""
        optimizer_name = self.config.get('optimizer', 'sgd').lower()
        lr = self.config.get('learning_rate', 0.01)
        weight_decay = self.config.get('weight_decay', 0.0005)

        if optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )

    def _create_scheduler(self, optimizer: torch.optim.Optimizer,
                          epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        """创建学习率调度器"""
        scheduler_name = self.config.get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )