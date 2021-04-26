"""
  @Author       : liujianhan
  @Date         : 2018/6/2 上午11:59
  @Project      : posture_classify
  @FileName     : data_processor.py
  @Description  : Placeholder
"""
import os
from typing import Tuple

import pandas as pd
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
import numpy as np


def data_generator_flow(_train_dir: str,
                        _valid_dir: str,
                        _test_dir: str,
                        batch_size: int = 32,
                        target_size: Tuple = (256, 256),
                        multi_output_mode: bool = False) -> Tuple:
    """
    数据生成器函数
    @param _train_dir: 训练数据文件路径
    @param _valid_dir: 验证数据文件路径
    @param _test_dir: 测试数据文件路径
    @param batch_size: 批量参数
    @param target_size: 目标转换形状
    @param multi_output_mode: 多输出模式
    @return: 生成器元组
    """
    train_df = pd.read_csv(os.path.join(_train_dir, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(_valid_dir, 'valid.csv'))
    test_df = pd.read_csv(os.path.join(_test_dir, 'test.csv'))
    if not multi_output_mode:
        train_df.label = train_df.label.astype('str')
        valid_df.label = valid_df.label.astype('str')
        test_df.label = test_df.label.astype('str')
    train_data_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        width_shift_range=.2,
        height_shift_range=.2,
        shear_range=.2,
        zoom_range=.2,
        channel_shift_range=np.random.choice(100),
        horizontal_flip=True,

    )
    train_data_flow = train_data_gen.flow_from_dataframe(
        dataframe=train_df,
        target_size=target_size,
        directory=_train_dir,
        batch_size=batch_size,
        class_mode='multi_output' if multi_output_mode else 'binary',
        x_col='filename',
        y_col=['label', 'score'] if multi_output_mode else 'label',
    )
    # 验证集不要做数据增强
    valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    valid_data_flow = valid_data_gen.flow_from_dataframe(
        dataframe=valid_df,
        target_size=target_size,
        directory=_valid_dir,
        batch_size=batch_size,
        class_mode='multi_output' if multi_output_mode else 'binary',
        x_col='filename',
        y_col=['label', 'score'] if multi_output_mode else 'label',
    )

    test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_data_flow = test_data_gen.flow_from_dataframe(
        dataframe=test_df,
        target_size=target_size,
        directory=_test_dir,
        batch_size=batch_size,
        class_mode='multi_output' if multi_output_mode else 'binary',
        x_col='filename',
        y_col=['label', 'score'] if multi_output_mode else 'label',
    )

    return train_data_flow, valid_data_flow, test_data_flow
