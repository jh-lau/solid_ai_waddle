"""
  @Author       : liujianhan
  @Date         : 2018/6/1 下午4:57
  @Project      : posture_classify
  @FileName     : train.py
  @Description  : Placeholder
"""
import logging
import os
import time
from datetime import datetime
from typing import List, Any

from dotmap import DotMap
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import multi_gpu_model

from ..layer.model import build_model
from ..ops.data_processor import data_generator_flow
from ..ops.utils import remove_file, get_last_create_file


def callback_functions(config_dict: DotMap) -> List:
    """
    创建模型训练的回调函数
    @param config_dict: 配置字典
    @return: 回调函数列表
    """
    logging = TensorBoard(log_dir=config_dict.tensorboard_logger_dir)
    checkpoint = ModelCheckpoint(os.path.join(config_dict.checkpoint_dir, config_dict.checkpoint_model_name),
                                 monitor='accuracy',
                                 save_weights_only=False,
                                 save_best_only=True,
                                 period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='accuracy', patience=3, mode='max')
    time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
    csv_logger = CSVLogger(os.path.join(config_dict.csv_logger_dir, f'csv_logger_{time_str}.csv'))

    return [logging, checkpoint, reduce_lr, early_stopping, csv_logger]


def train_customize_layer(callback_function_list: List,
                          config_dict: DotMap,
                          model: Any,
                          train_data_gen: Any,
                          valid_data_gen: Any,
                          test_data_gen: Any,
                          initial_epoch: int,
                          target_layer: str,
                          eval_flag: bool = True,
                          multi_output_mode: bool = False,
                          use_pretrain_weights: bool = False,
                          use_multi_gpus: bool = True,
                          gpus_num: int = 2) -> Any:
    """
    训练模型的特定层方法
    @param callback_function_list: 回调函数列表
    @param config_dict: 配置参数字典
    @param model: 模型
    @param train_data_gen: 训练数据生成器
    @param valid_data_gen: 验证数据生成器
    @param test_data_gen: 测试数据生成器
    @param initial_epoch: 开始训练的迭代数，如果是从已有模型继续训练，一般不为0
    @param target_layer: 目标层名称，在该层之后的所有层均会被重新训练
    @param eval_flag: 是否进行评估
    @param multi_output_mode: 是否多输出模式
    @param use_pretrain_weights: 是否使用预训练权重
    @param use_multi_gpus: 是否多gpu训练
    @param gpus_num: gpu数量
    @return:
    """
    # 自定义训练层
    if use_pretrain_weights:
        train_flag = False
        for layer in model.layers:
            if layer.name == target_layer:
                train_flag = True
            if layer.name.startswith('bn'):
                layer.trainable = True
            if train_flag:
                layer.trainable = True
            else:
                layer.trainable = False
    logging.info(f"Model's trainable weights number is: {len(model.trainable_weights)}")

    if use_multi_gpus:
        model = multi_gpu_model(model, gpus=gpus_num)
    if multi_output_mode:
        model.compile(optimizer=Adam(lr=config_dict.learning_rate),
                      loss={'action_output': 'binary_crossentropy',
                            'score_output': 'sparse_categorical_crossentropy'},
                      loss_weights={'action_output': .8,
                                    'score_output': .5},
                      metrics={'action_output': 'accuracy',
                               'score_output': 'accuracy'})
    else:
        model.compile(optimizer=Adam(lr=config_dict.learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    model.fit_generator(
        use_multiprocessing=True,
        workers=10,
        shuffle=False if use_pretrain_weights else True,
        generator=train_data_gen,
        validation_data=valid_data_gen,
        epochs=config_dict.epoch,
        verbose=1,
        initial_epoch=initial_epoch,
        callbacks=callback_function_list
    )

    if eval_flag:
        logging.info(f"Evaluating on test data.....")
        res = model.evaluate_generator(generator=test_data_gen,
                                       use_multiprocessing=True,
                                       workers=10,
                                       verbose=1)
        for metric, r in zip(model.metrics_names, res):
            logging.info(f"{metric}: {r:.3f}")

    return model


def train_model(config_dict: DotMap,
                continue_training: bool = False,
                remove_tensorboard_logger: bool = True,
                multi_output_mode: bool = False,
                use_pretrain_weights: bool = False,
                use_multi_gpus: bool = True,
                gpus_num: int = 2) -> None:
    """
    训练模型函数
    @param config_dict: 配置字典文件
    @param continue_training: 是否加载checkpoint文件继续训练
    @param remove_tensorboard_logger: 是否清空旧日志文件
    @param multi_output_mode: 是否多输出模式
    @param use_pretrain_weights:
    @param use_multi_gpus: 是否多gpu训练
    @param gpus_num: gpu数量
    @return:
    """
    logging.info(f"Training start at: {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    start_timestamp = time.time()
    train_data_gen, valid_data_gen, test_data_gen = data_generator_flow(config_dict.train_data_dir,
                                                                        config_dict.valid_data_dir,
                                                                        config_dict.test_data_dir,
                                                                        config_dict.batch_size,
                                                                        multi_output_mode=multi_output_mode)

    if remove_tensorboard_logger:
        remove_file(config_dict.tensorboard_logger_dir)
    last_ckpt_file = get_last_create_file(config_dict.checkpoint_dir)
    initial_epoch = 0
    if continue_training and last_ckpt_file:
        initial_epoch = int(os.path.basename(last_ckpt_file)[2:5])
        model = load_model(last_ckpt_file)
        logging.info(f"Last checkpoint file {last_ckpt_file} was loaded.")
    else:
        model = build_model(tuple(config_dict.input_image_shape),
                            config_dict.imagenet_path,
                            multi_output_mode=multi_output_mode)

    callback_function_list = callback_functions(config_dict)

    if use_pretrain_weights:
        # 训练自定义输出层
        model = train_customize_layer(callback_function_list,
                                      config_dict,
                                      model,
                                      train_data_gen,
                                      valid_data_gen,
                                      test_data_gen,
                                      initial_epoch,
                                      target_layer='action_output',
                                      use_pretrain_weights=use_pretrain_weights,
                                      multi_output_mode=multi_output_mode,
                                      use_multi_gpus=use_multi_gpus,
                                      gpus_num=gpus_num)

        # 微调原有模型+自定义层
        initial_epoch = 0
        model = train_customize_layer(callback_function_list,
                                      config_dict,
                                      model,
                                      train_data_gen,
                                      valid_data_gen,
                                      test_data_gen,
                                      initial_epoch,
                                      target_layer='activation_40',
                                      multi_output_mode=multi_output_mode,
                                      use_pretrain_weights=use_pretrain_weights,
                                      use_multi_gpus=use_multi_gpus,
                                      gpus_num=gpus_num)
    else:
        model = train_customize_layer(callback_function_list,
                                      config_dict,
                                      model,
                                      train_data_gen,
                                      valid_data_gen,
                                      test_data_gen,
                                      initial_epoch,
                                      target_layer='',
                                      multi_output_mode=multi_output_mode,
                                      use_pretrain_weights=use_pretrain_weights,
                                      use_multi_gpus=use_multi_gpus,
                                      gpus_num=gpus_num)
    time_tag = datetime.now().strftime('%Y%m%d%H%M')
    model_name = f'{os.path.join(config_dict.final_model_dir, time_tag)}.h5'
    try:
        model.save(model_name)
    except OSError:
        model.save(f'{config_dict.final_model_dir}/latest_model.h5')
    logging.info('Model saved.')
    logging.info(f"Training end at: {time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}")
    logging.info(f"Training time cost: {(time.time() - start_timestamp)/60:.3f} min")


if __name__ == '__main__':
    print()
