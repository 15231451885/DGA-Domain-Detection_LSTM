"""
模型训练模块（对抗样本版本）
"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from Config import Config
from DataPreprocess import DataPreprocess
from Model import DGA_Detection_Model
from Utils_adv import DataLoader
from AdversarialGenerator_adv import AdversarialGenerator


def train():
    """训练主函数"""
    # 初始化配置
    config = Config()
    
    # 数据预处理
    preprocessor = DataPreprocess(config)
    train_examples, val_examples = preprocessor.load_data()
    
    # 初始化对抗样本生成器
    adv_generator = AdversarialGenerator(perturbation_strength=0.3)
    data_loader = DataLoader(adversarial_generator=adv_generator)
    
    # 加载批次数据（会自动应用对抗样本）
    train_data = data_loader.load_batch(train_examples, dataset="train")
    val_data = data_loader.load_batch(val_examples, dataset="val")
    
    # 模型训练
    model = DGA_Detection_Model(config).build_model()
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath=os.path.join(config.model_save_dir, "model_adv.h5"),
            save_best_only=True
        )
    ]
    
    history = model.fit(
        x=np.array([example.domain_name for example in train_data]),
        y=np.array([example.label for example in train_data]),
        validation_data=(
            np.array([example.domain_name for example in val_data]),
            np.array([example.label for example in val_data])
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=callbacks
    )
    
    print(f"训练完成，模型已保存到: {os.path.join(config.model_save_dir, 'model_adv.h5')}")


if __name__ == "__main__":
    train()