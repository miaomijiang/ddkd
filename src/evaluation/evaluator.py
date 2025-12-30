"""
模型评估器
"""
import torch
import torch.nn as nn
import time
from typing import Dict


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, device: torch.device):
        self.device = device

    def evaluate(self, model: nn.Module,
                 test_loader: torch.utils.data.DataLoader) -> Dict:
        """全面评估模型"""
        model.eval()

        # 准确率评估
        accuracy_metrics = self._evaluate_accuracy(model, test_loader)

        # 推理时间评估
        time_metrics = self._evaluate_inference_time(model, test_loader)

        # 模型复杂度
        complexity_metrics = self._evaluate_complexity(model)

        # 合并结果
        return {**accuracy_metrics, **time_metrics, **complexity_metrics}

    def _evaluate_accuracy(self, model: nn.Module,
                           test_loader: torch.utils.data.DataLoader) -> Dict:
        """评估准确率"""
        model.eval()

        total_correct_top1 = 0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                total_samples += labels.size(0)

                # Top-1准确率
                _, predicted = outputs.max(1)
                total_correct_top1 += predicted.eq(labels).sum().item()

                # Top-5准确率
                _, top5_pred = outputs.topk(5, 1, True, True)
                total_correct_top5 += top5_pred.eq(labels.view(-1, 1)).sum().item()

        accuracy_top1 = 100. * total_correct_top1 / total_samples
        accuracy_top5 = 100. * total_correct_top5 / total_samples

        return {
            'accuracy_top1': accuracy_top1,
            'accuracy_top5': accuracy_top5
        }

    def _evaluate_inference_time(self, model: nn.Module,
                                 test_loader: torch.utils.data.DataLoader) -> Dict:
        """评估推理时间"""
        model.eval()

        # 预热
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        for _ in range(10):
            _ = model(dummy_input)

        # 测量推理时间
        total_time = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                batch_size = images.size(0)

                start_time = time.perf_counter()
                _ = model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()

                total_time += (end_time - start_time)
                num_batches += 1

                if num_batches >= 20:
                    break

        avg_batch_time = total_time / num_batches if num_batches > 0 else 0
        avg_sample_time = avg_batch_time / batch_size if batch_size > 0 else 0

        return {
            'inference_time_per_sample': avg_sample_time,
            'inference_time_per_batch': avg_batch_time
        }

    def _evaluate_complexity(self, model: nn.Module) -> Dict:
        """评估模型复杂度"""
        from thop import profile

        # 计算FLOPs和参数量
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 计算模型大小
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        model_size_mb = param_size / 1024 ** 2

        return {
            'flops': flops,
            'total_parameters': params,
            'model_size_mb': model_size_mb
        }

    def compare_models(self, baseline_model: nn.Module,
                       distilled_model: nn.Module,
                       test_loader: torch.utils.data.DataLoader) -> Dict:
        """比较两个模型"""
        baseline_results = self.evaluate(baseline_model, test_loader)
        distilled_results = self.evaluate(distilled_model, test_loader)

        comparison = {
            'baseline': baseline_results,
            'distilled': distilled_results,
            'improvement': {
                'accuracy_top1': distilled_results['accuracy_top1'] - baseline_results['accuracy_top1'],
                'accuracy_top5': distilled_results['accuracy_top5'] - baseline_results['accuracy_top5'],
                'inference_time_ratio': distilled_results['inference_time_per_sample'] / baseline_results[
                    'inference_time_per_sample'],
                'size_ratio': distilled_results['model_size_mb'] / baseline_results['model_size_mb']
            }
        }

        return comparison