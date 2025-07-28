# 导入模块
from Utils import DataLoader, init_logger, plot_history, conclusion  # 数据加载、日志初始化、历史记录绘图和结论生成
from Config import Param  # 配置文件
import torch  # PyTorch深度学习框架
from typing import Tuple, Generator  # 类型注解
from Model import ClassfierUsingLstm  # LSTM分类模型
from tqdm import tqdm  # 进度条
import torch.optim as optim  # 优化器
import json  # JSON处理
import numpy as np  # 数值计算
import os  # 操作系统接口

# 初始化日志记录器
logger = init_logger("train", Param.log_dir)
# 设置设备（CPU或GPU）
DEVICE = torch.device(Param.device)

# 计算二分类任务的评估指标（精确率、召回率、F1、准确率）
def BinarySeqLabel(pred: torch.tensor, truth: torch.tensor) -> Tuple:
    # 计算真阳性（TP）、真阴性（TN）、假阴性（FN）、假阳性（FP）
    TP = ((pred == 1) & (truth == 1)).cpu().sum().item()
    TN = ((pred == 0) & (truth == 0)).cpu().sum().item()
    FN = ((pred == 0) & (truth == 1)).cpu().sum().item()
    FP = ((pred == 1) & (truth == 0)).cpu().sum().item()

    # 计算精确率（Precision）
    p = 0 if TP == FP == 0 else TP / (TP + FP)
    # 计算召回率（Recall）
    r = 0 if TP == FN == 0 else TP / (TP + FN)
    # 计算F1分数
    F1 = 0 if p == r == 0 else 2 * r * p / (r + p)
    # 计算准确率（Accuracy）
    acc = (TP + TN) / (TP + TN + FP + FN)

    return (p, r, F1, acc)

# 线性学习率预热函数
def warmup_linear(step_ratio, warmup_ratio=0.002) -> float:
    if step_ratio < warmup_ratio:
        return step_ratio / warmup_ratio
    return 1.0 - step_ratio

# 训练函数
def train(model: torch.nn.Module,
         train_epoch_num: int,
         train_batch_num_one_epoch: int,
         train_batch_iter: Generator,
         train_batch_size: int,
         valid_batch_iter: Generator,
         valid_batch_num_one_epoch: int,
         valid_batch_size: int):

    # 将模型移动到指定设备（CPU或GPU）
    model = model.to(DEVICE)
    logger.info("__________________start_training___________________")

    # 初始化优化器（Adam）
    optimizer = optim.Adam(model.parameters(), lr=Param.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # 启用CUDA同步调试

    # 初始化训练和验证的历史记录
    T_losses, T_p, T_r, T_f1, T_acc = [], [], [], [], []
    V_losses, V_p, V_r, V_f1, V_acc = [], [], [], [], []
    history = {
        "train_loss": T_losses,
        "train_precision": T_p,
        "train_recall": T_r,
        "train_f1": T_f1,
        "train_acc": T_acc,
        "dev_loss": V_losses,
        "dev_precision": V_p,
        "dev_recall": V_r,
        "dev_f1": V_f1,
        "dev_acc": V_acc
    }

    global_step = 0
    t_total = train_epoch_num * train_batch_num_one_epoch  # 总训练步数

    # 训练循环
    for epoch in range(train_epoch_num):
        train_acc_num = 0
        pbar = tqdm(total=train_batch_num_one_epoch)  # 进度条
        losses, p, r, f1, acc = [], [], [], [], []

        # 批次训练
        for batch_idx in range(train_batch_num_one_epoch):
            optimizer.zero_grad()  # 清空梯度
            batch = train_batch_iter.__next__()  # 获取批次数据
            labels = batch.labels.to(DEVICE)  # 标签移动到设备
            prob = model(batch)  # 模型预测
            predict = prob.argmax(dim=1)  # 预测结果
            loss = model.loss_fn(prob, labels)  # 计算损失

            # 计算评估指标
            b_p, b_r, b_f1, b_acc = BinarySeqLabel(predict.cpu(), labels.cpu())
            p.append(b_p)
            r.append(b_r)
            f1.append(b_f1)
            acc.append(b_acc)

            # 统计正确预测数
            acc_num = (predict == labels).sum().item()
            train_acc_num += acc_num

            losses.append(loss.item())  # 记录损失
            loss.backward()  # 反向传播

            # 更新学习率（线性预热）
            lr_this_step = Param.learning_rate * warmup_linear(global_step / t_total, Param.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空梯度
            global_step += 1  # 全局步数增加

            # 更新进度条
            pbar.set_description(
                'Epoch: %2d (learing rate : %.6f )| LOSS: %2.3f | F1: %1.3f | ACC: %1.3f| PRECISION: %1.3f | RECALL: %1.3f' % (
                    epoch, lr_this_step, loss.item(), b_f1, b_acc, b_p, b_r))
            pbar.update(1)
            optimizer.step()

        # 计算并记录训练指标
        train_acc = train_acc_num / (train_batch_size * train_batch_num_one_epoch)
        losses, p, r, f1, acc = list(map(np.mean, [losses, p, r, f1, acc]))
        logger.info('Train Epoch: %2d  (learing rate : %.4f )| LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, lr_this_step, losses, f1, train_acc, p, r))

        T_p.append(p)
        T_r.append(r)
        T_f1.append(f1)
        T_acc.append(train_acc)
        T_losses.append(losses)

        pbar.clear()
        pbar.close()

        # 验证循环
        valid_acc_num = 0
        losses, p, r, f1, acc = [], [], [], [], []

        with torch.no_grad():
            for batch_idx in range(valid_batch_num_one_epoch):
                batch = valid_batch_iter.__next__()
                labels = batch.labels.to(DEVICE)
                prob = model(batch)
                predict = prob.argmax(dim=1)
                loss = model.loss_fn(prob, labels)
                b_p, b_r, b_f1, b_acc = BinarySeqLabel(predict.cpu(), labels.cpu())
                p.append(b_p)
                r.append(b_r)
                f1.append(b_f1)
                acc.append(b_acc)
                losses.append(loss.item())
                acc_num = (predict == labels).sum().item()
                valid_acc_num += acc_num

        # 计算并记录验证指标
        valid_acc = valid_acc_num / (valid_batch_size * valid_batch_num_one_epoch)
        losses, p, r, f1, acc = list(map(np.mean, [losses, p, r, f1, acc]))
        logger.info('Valid Epoch: %2d | LOSS: %2.3f | F1: %1.3f | ACC: %1.3f | PRECISION: %1.3f | RECALL: %1.3f' %
                    (epoch, losses, f1, valid_acc, p, r))
        V_acc.append(valid_acc)
        V_p.append(p)
        V_r.append(r)
        V_f1.append(f1)
        V_losses.append(losses)

        # 保存历史记录
        with open(Param.log_dir + "History.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(history))

        # 绘制历史记录图
        if len(V_acc) >= 2:
            plot_history(Param.log_dir + "History.json")

        # 保存最佳模型
        if len(V_acc) == 1 or V_f1 >= max(V_f1[:-1]):
            model_to_save = model.module if hasattr(model, 'module') else model  # 仅保存模型本身
            output_model_file = os.path.join(Param.cached_dir, "pytorch_model_%s.bin" % epoch)
            torch.save(model_to_save.state_dict(), output_model_file)


# 加载词汇表
vocab = torch.load(Param.data_dir + "vocab.pt")
# 初始化数据加载器
dataloader = DataLoader(
    train_examples_dir=Param.data_dir + "train_examples.pt",
    valid_examples_dir=Param.data_dir + "valid_examples.pt",
    vocab=vocab
)
# 计算动态权重
weight = torch.tensor([1.0, Param.calculate_class_weights(Param.train_data_path)], dtype=torch.float)
# 初始化LSTM分类模型（传入动态权重）
model = ClassfierUsingLstm(
    vocab_size=vocab.size,
    char_embedding_dim=Param.char_embedding_dim,
    hidden_size=Param.hidden_size,
    num_layers=Param.num_layers,
    bidirectional=Param.bidirectional,
    dense_size=Param.dense_size,
    weight=weight
)
# 启动训练
train(
    model=model,
    train_epoch_num=Param.train_epoch_num,
    train_batch_num_one_epoch=Param.train_batch_num_one_epoch,
    train_batch_iter=dataloader.Get_Batch_Iter(dataset="train", batch_size=Param.train_batch_size, show_example=True),
    train_batch_size=Param.train_batch_size,
    valid_batch_iter=dataloader.Get_Batch_Iter(dataset="valid", batch_size=Param.valid_batch_size, show_example=False),
    valid_batch_num_one_epoch=Param.valid_batch_num_one_epoch,
    valid_batch_size=Param.valid_batch_size
)
# 生成训练结论
conclusion(Param.log_dir + "History.json")