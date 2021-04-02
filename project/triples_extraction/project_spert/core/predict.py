"""
  @Author       : liujianhan
  @Date         : 2018/3/3 下午5:11
  @Project      : triples_extraction
  @FileName     : predict.py
  @Description  : Placeholder
"""
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from .spert import sampling, util
from .spert.entities import Dataset
from .spert.evaluator import Evaluator
from .spert.input_reader import JsonInputReader
from ..ops.utils import split_chi_char


def spo_predict(sentence: str,
                config_args: dict,
                tokenizer: Any,
                spert_model: Any,
                device: Any) -> dict:
    """
    预测spo三元组函数
    :param sentence: 目标句子
    :param config_args: 配置参数字典
    :param tokenizer: BERT分词器
    :param spert_model: 预训练模型
    :param device: 预测环境：GPU/CPU
    :return: 预测结果：实体以及实体关系
    """
    input_reader = JsonInputReader(config_args['type_path'],
                                   tokenizer,
                                   max_span_size=config_args['max_span_size'])
    split_sentence = split_chi_char(sentence)
    target_dataset = {'tokens': split_sentence,
                      'entities': [],
                      'relations': []}
    input_reader.read({'test': target_dataset})
    final_dataset = input_reader.get_dataset('test')

    evaluator = Evaluator(dataset=final_dataset,
                          input_reader=input_reader,
                          text_encoder=tokenizer,
                          rel_filter_threshold=config_args['rel_filter_threshold'],
                          predictions_path='',
                          examples_path='',
                          example_count=None,
                          epoch=0,
                          dataset_label='test')
    final_dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(final_dataset,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False,
                             num_workers=4,
                             collate_fn=sampling.collate_fn_padding)
    with torch.no_grad():
        spert_model.eval()

        total = math.ceil(final_dataset.document_count)
        # for batch in tqdm(data_loader, total=total, desc='Prediction of Sentence:'):
        for batch in data_loader:
            batch = util.to_device(batch, device)

            result = spert_model(encodings=batch['encodings'],
                                 context_masks=batch['context_masks'],
                                 entity_masks=batch['entity_masks'],
                                 entity_sizes=batch['entity_sizes'],
                                 entity_spans=batch['entity_spans'],
                                 entity_sample_masks=batch['entity_sample_masks'],
                                 evaluate=True)
            entity_clf, rel_clf, relations = result

            evaluator.eval_batch(entity_clf, rel_clf, relations, batch)

    pred_entities, pred_relations = evaluator.get_prediction()
    result_dict = {"text": sentence, "spo_list": []}

    for pr in pred_relations:
        for spo in pr:
            spo_dict = {'subject': ''.join(split_sentence[spo[0][0] - 1:spo[0][1] - 1]),
                        'predicate': spo[2].identifier,
                        'object': ''.join(split_sentence[spo[1][0] - 1:spo[1][1] - 1]),
                        'subject_type': spo[0][2].identifier,
                        'object_type': spo[1][2].identifier}
            if spo_dict not in result_dict['spo_list']:
                result_dict['spo_list'].append(spo_dict)

    return result_dict
