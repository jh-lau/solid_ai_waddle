"""
  @Author       : liujianhan
  @Date         : 2020/5/15 上午10:52
  @Project      : KGE
  @FileName     : utils.py
  @Description  : 数据处理公用函数
"""
import logging
import os
from typing import List

from tqdm import tqdm
import random


def set_logger(save_file: str) -> None:
    """
    设置训练输出和输出保存文件
    @param save_file: 保存的输出信息文件：如test.log
    @return:
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        filename=save_file,
        filemode='w',
        level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def get_entities_relations_dict(triple_path: str,
                                target_dir: str,
                                separator: str,
                                sample: int,
                                ratio: List[float] = [.8, .1, .1]) -> None:
    """
    从数据三元组中抽取符合模型输入的字段保存成各自文件
    @param triple_path: 三元组语料路径
    @param target_dir: 保存文件目标路径
    @param separator: 语料三元组的分隔符，如\t或者;
    @param sample: 从语料中抽取的三元组数量
    @param ratio: 对三元组进行训练集、开发集和测试集的切分比例
    @return:
    """
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    entities, relations, triple_sample = {}, {}, []
    with open(triple_path, 'r', encoding='utf-8') as f:
        for index, line in tqdm(enumerate(f), total=sample):
            if index > sample:
                break

            tmp_result = [s.replace(' ', '').replace('\xa0', '').replace('\u3000', '')
                          for s in line.strip().split(separator, 2) if s]
            if len(tmp_result) == 3:
                triple_sample.append('\t'.join(tmp_result))
                head, relation, tail = tmp_result
                entities[head] = entities.get(head, len(entities))
                entities[tail] = entities.get(tail, len(entities))
                relations[relation] = relations.get(relation, len(relations))
    entities_file = os.path.join(target_dir, 'entities.dict')
    relations_file = os.path.join(target_dir, 'relations.dict')
    train_file = os.path.join(target_dir, 'train.txt')
    valid_file = os.path.join(target_dir, 'valid.txt')
    test_file = os.path.join(target_dir, 'test.txt')

    logging.info(f'Entity numbers: {len(entities)}')
    logging.info(f'Relation numbers: {len(relations)}')

    for dic, file in zip([entities, relations], [entities_file, relations_file]):
        with open(file, 'w', encoding='utf-8') as f:
            for k, v in dic.items():
                f.writelines(f'{v}\t{k}\n')

    random.shuffle(triple_sample)
    train_sample = triple_sample[:int(sample * ratio[0])]
    valid_sample = triple_sample[int(sample * ratio[0]): int(sample * (ratio[0] + ratio[1]))]
    test_sample = triple_sample[int(sample * (ratio[0] + ratio[1])):]

    logging.info(f'Train datasets: {len(train_sample)}')
    logging.info(f'Valid datasets: {len(valid_sample)}')
    logging.info(f'Test datasets: {len(test_sample)}')

    for s, file in zip([train_sample, valid_sample, test_sample], [train_file, valid_file, test_file]):
        with open(file, 'w', encoding='utf-8') as f:
            for content in s:
                f.writelines(f'{content}\n')


if __name__ == '__main__':
    set_logger()
    data_path = '../data/cn_military/triples.txt'
    target_path = '../data/cn_military_300k'
    get_entities_relations_dict(data_path, target_path, sample=78500, separator='\t')
