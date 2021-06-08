"""
  @Author       : liujianhan
  @Date         : 2018/6/1 下午4:57
  @Project      : posture_classification
  @FileName     : model.py
  @Description  : Placeholder
"""
from typing import Tuple, Any

from keras import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.layers import Dense


def build_model(input_image_shape: Tuple,
                imagenet_path: str,
                multi_output_mode: bool = False,
                use_pretrain_weights: bool = False) -> Any:
    """
    构建多输出的基于ResNet50的带预训练权重模型
    @param input_image_shape: 输入图片形状
    @param imagenet_path: 预训练权重路径
    @param multi_output_mode: 多输出模式
    @param use_pretrain_weights: 是否使用预训练权重
    @return: 模型架构
    """
    conv_base = ResNet50(weights=imagenet_path if use_pretrain_weights else None,
                         pooling='avg',
                         include_top=False,
                         input_shape=input_image_shape)
    action_output = Dense(1, activation='sigmoid', name='action_output')(conv_base.output)
    if multi_output_mode:
        score_output = Dense(3, activation='softmax', name='score_output')(conv_base.output)
        output = [action_output, score_output]
    else:
        output = [action_output]
    model = Model(conv_base.input, output, name='multi_task_model' if multi_output_mode else 'single_task_model')

    return model
