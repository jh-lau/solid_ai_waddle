"""
  @Author       : liujianhan
  @Date         : 2018/5/26 上午10:12
  @Project      : posture_classify
  @FileName     : service.py
  @Description  : service服务接口模块
"""
import os
import sys
from typing import Union

import cv2
import numpy as np
from PIL import Image
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model as keras_load_model

from .core.train import train_model
from .ops.utils import get_dot_config, set_logger

config_dic = get_dot_config('action_class/config.yml')
model = None


def load_model():
    global model
    model = keras_load_model(config_dic.final_model_file)


def inference(image_input: Union[str, np.array]):
    if model is not None:
        h, w = config_dic.input_image_shape[:2]
        if isinstance(image_input, str):
            image_data = Image.open(image_input)
            image_data = np.array(image_data.resize((h, w), Image.NEAREST))
        else:
            image_input_processed = preprocess_input(image_input)
            image_data = cv2.resize(image_input_processed, (h, w), interpolation=cv2.INTER_NEAREST)
        image_data = np.expand_dims(image_data, axis=0)
        action, rank = model.predict(image_data)
        score_dict = {0: np.random.randint(50),
                      1: np.random.randint(50, 75),
                      2: np.random.randint(75, 100)}
        score = score_dict[np.argmax(rank)]
        prob = 1 if action.item() > .5 else 0

        return image_input, prob, score
    else:
        print('模型未正确加载，请确认模型文件相关配置信息。')


if __name__ == '__main__':
    set_logger(config_dic.output_stream_logger_file)
    import argparse

    cmd_parser = argparse.ArgumentParser(description="Train or Inference")
    cmd_parser.add_argument('--train', action='store_true', default=False)
    cmd_args = cmd_parser.parse_args()
    if cmd_args.train:
        train_model(config_dic)
    else:
        image_path = 'data_path/data/train/images/positive_3/frame_000001.jpg'
        load_model()
        res = inference(image_path)
        print(res)
