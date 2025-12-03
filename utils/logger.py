from torch.utils.tensorboard import SummaryWriter
from config.config import *


class TensorboardLogger:
    def __init__(self):
        self.writer = SummaryWriter(log_dir=LOG_DIR)

    def log_train(self, loss_dict, acc, step):
        """记录训练数据（loss、acc）"""
        self.writer.add_scalar("Train/Total_Loss", loss_dict["total"], step)
        self.writer.add_scalar("Train/Cls_Loss", loss_dict["cls"], step)
        self.writer.add_scalar("Train/Distill_Loss", loss_dict["distill"], step)
        self.writer.add_scalar("Train/Gradient_Match_Loss", loss_dict["grad"], step)
        self.writer.add_scalar("Train/Accuracy", acc, step)

    def log_val(self, loss, acc, epoch):
        """记录验证数据（loss、acc）"""
        self.writer.add_scalar("Val/Total_Loss", loss, epoch)
        self.writer.add_scalar("Val/Accuracy", acc, epoch)

    def close(self):
        self.writer.close()