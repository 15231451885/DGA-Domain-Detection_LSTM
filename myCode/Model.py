import torch
from DataStructure import BatchInputFeature
from typing import Dict
from Config import Param
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Utils import init_logger
import torch.nn.functional as F

# 初始化日志记录器，用于记录训练过程中的信息
logger = init_logger("train",Param.log_dir)

# 设置设备（CPU或GPU）
DEVICE = torch.device(Param.device)

# Focal Loss 实现
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 基于LSTM的分类器模型
class ClassfierUsingLstm(torch.nn.Module):

	def __init__(self,
				 vocab_size : int ,
				 char_embedding_dim : int,
				 hidden_size : int,
				 num_layers : int,
				 bidirectional : bool,
				 dense_size : int,
				 weight : torch.tensor = None
				 ):
		"""
		初始化LSTM分类器
		参数:
			vocab_size: 词汇表大小
			char_embedding_dim: 字符嵌入维度
			hidden_size: LSTM隐藏层维度
			num_layers: LSTM层数
			bidirectional: 是否使用双向LSTM
			dense_size: 全连接层维度
		"""
		super().__init__()
		# 字符嵌入层，将字符ID映射为向量
		self.char_embedding = torch.nn.Embedding(
			vocab_size + 5 ,char_embedding_dim,padding_idx = 0
		)
		# 是否双向LSTM
		self.bidirectional = bidirectional
		# LSTM编码器
		self.encoder = torch.nn.LSTM(
			input_size = char_embedding_dim,
			hidden_size = hidden_size,
			num_layers = num_layers,
		    bidirectional = bidirectional)

		# 编码器输出维度（双向LSTM时维度加倍）
		encoder_dim = (bidirectional + 1) * hidden_size
		# 全连接层，用于分类
		self.dense = torch.nn.Sequential(
			torch.nn.Linear(encoder_dim,dense_size),
			torch.nn.Tanh(),
			torch.nn.Linear(dense_size,2),
			torch.nn.Softmax(dim = -1)
		)
		# 损失函数（Focal Loss，alpha=0.25, gamma=2，支持动态权重）
		self.loss = FocalLoss(alpha=0.25, gamma=2, weight=weight)

	def forward(self, batch : BatchInputFeature ) -> torch.tensor :
		"""
		前向传播
		参数:
			batch: 输入批次数据
		返回:
			prob: 分类概率
		"""
		# 字符嵌入
		char_embedding = self.char_embedding(batch.char_ids.to(DEVICE))
		# 打包变长序列
		char_embedding_packed = pack_padded_sequence(
			input = char_embedding,
			batch_first = True,
			lengths = batch.domain_lens,
			enforce_sorted = False
		)
		# LSTM编码
		_,(hn,cn) = self.encoder(char_embedding_packed)
		# 处理双向LSTM的输出
		if not self.bidirectional:
			batch_domain_encoded = hn[-1]
		else:
			batch_domain_encoded = torch.cat(
				[hn[-1],hn[-2]],dim = 1
			)
		# 分类概率
		prob = self.dense(batch_domain_encoded)
		return prob

	def loss_fn(self,prob : torch.tensor ,label : torch.tensor ):
		"""
		计算损失
		参数:
			prob: 预测概率
			label: 真实标签
		返回:
			loss: 损失值
		"""
		return self.loss(prob,label).cpu()

	def from_pretrained(self,cached_path_dir, mode = "eval"):
		"""
		加载预训练模型
		参数:
			cached_path_dir: 模型路径
			mode: 模式（eval或train）
		返回:
			self: 加载后的模型
		"""
		logger.info("load pre_trained model from %s " % cached_path_dir)
		model_state_dict = torch.load(cached_path_dir, map_location=DEVICE)
		self.load_state_dict(model_state_dict)
		if mode == "eval":
			self.eval()
		if mode == "train":
			self.train()
		return self
