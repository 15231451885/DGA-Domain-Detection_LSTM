import os
from typing import Dict
import numpy as np

class Param:
    #path configuration
    device: str = "cpu"
    root: str = os.getcwd()
    data_dir: str = os.path.join(root, "data/")
    train_data_path: str = os.path.join(data_dir, "train_examples.pt")  # 训练数据路径
    valid_data_path: str = os.path.join(data_dir, "valid_examples.pt")  # 验证数据路径
    log_dir: str = os.path.join(root, "log/")
    cached_dir: str = os.path.join(root, "model_saved/")

    #Model
    char_embedding_dim: int = 128
    hidden_size: int = 100
    num_layers: int = 2
    bidirectional: bool = True
    dense_size: int = 100
    # 动态计算类别权重（默认值为5.0，可运行时覆盖）
    positive_sample_loss_weight: float = 5.0  # 默认值
    
    @staticmethod
    def calculate_class_weights(train_data_path: str) -> float:
        """动态计算DGA样本的损失权重"""
        try:
            import torch
            data = torch.load(train_data_path)
            if isinstance(data, tuple):
                labels = data[1]  # 假设标签在第二个维度
            else:
                labels = data  # 如果数据直接是标签
            
            if isinstance(labels, torch.Tensor):
                positive_count = (labels == 1).sum().item()
            else:
                positive_count = sum(1 for label in labels if label == 1)
            
            negative_count = len(labels) - positive_count
            return max(1.0, positive_count / (negative_count + 1e-5))
        except (FileNotFoundError, RuntimeError):
            return Param.positive_sample_loss_weight  # 回退到默认值

    #Train
    learning_rate = 3e-4
    warmup_proportion = 0.05
    max_domain_len = 130

    train_epoch_num: int = 15
    train_batch_size: int = 2000
    train_batch_num_one_epoch: int = int(800000 / train_batch_size)

    valid_batch_size = 500
    valid_batch_num_one_epoch: int = int(200000 / valid_batch_size)
