"""
  @Author       : liujianhan
  @Date         : 2020/3/2 下午6:25
  @Project      : triples_extraction
  @FileName     : service.py
  @Description  : Placeholder
"""

import torch
from transformers import BertTokenizer

from .core.predict import spo_predict
from .core.spert import models
from .ops.utils import parse_yaml_parameters

CONFIG_ARGS = parse_yaml_parameters('pst/triples_extraction/parameters.yaml')
spert_model, tokenizer, device = None, None, None


def load_model(cpu: bool = True):
    """
    模型加载
    :param cpu: 是否使用cpu计算，默认为真
    :return:
    """
    global spert_model, tokenizer, device
    model_class = models.get_model('spert')
    tokenizer = BertTokenizer.from_pretrained(CONFIG_ARGS['model_path'])
    spert_model = model_class.from_pretrained(CONFIG_ARGS['model_path'],
                                              cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),
                                              relation_types=CONFIG_ARGS['relation_types'] - 1,
                                              entity_types=CONFIG_ARGS['entity_types'],
                                              max_pairs=CONFIG_ARGS['max_pairs'],
                                              prop_drop=CONFIG_ARGS['prop_drop'],
                                              size_embedding=CONFIG_ARGS['size_embedding'],
                                              freeze_transformer=False)
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    spert_model.to(device)


def inference(sentence: str):
    result_dict = spo_predict(sentence, CONFIG_ARGS, tokenizer,
                              spert_model, device)

    return result_dict
