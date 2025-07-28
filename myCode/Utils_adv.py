"""
数据加载工具模块
功能：
1. 加载训练/测试数据
2. 动态生成对抗样本（仅训练集）
3. 数据批处理
"""
import random
from typing import List

from DataStructure import DataExample


class DataLoader:
    def __init__(self, adversarial_generator=None):
        """
        :param adversarial_generator: 对抗样本生成器
        """
        self.adversarial_generator = adversarial_generator

    def load_batch(self, batch_examples: List[DataExample], dataset: str = "train"):
        """
        加载一个批次的数据（支持动态生成对抗样本）
        :param batch_examples: 原始数据样本
        :param dataset: 数据集类型（train/test）
        :return: 处理后的批次数据
        """
        if not batch_examples:
            return []

        # 动态生成对抗样本（仅对训练集且配置了生成器时生效）
        if dataset == "train" and self.adversarial_generator:
            adversarial_names = self.adversarial_generator.batch_generate(
                [example.domain_name for example in batch_examples],
                [example.label for example in batch_examples]  # 传递标签信息
            )
            for i, example in enumerate(batch_examples):
                example.domain_name = adversarial_names[i]

        return batch_examples