"""
  @Author       : liujianhan
  @Date         : 2020/3/5 下午2:03
  @Project      : tripples_extraction
  @FileName     : data_management.py
  @Description  : 标注数据预处理模块
"""
import codecs
import random
import time
from copy import deepcopy
from typing import Union, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import convert_train_valid_data_v2, \
    save_result_to_json, \
    convert_data_types


def manage_data(xls_file: str) -> list:
    """
    转换xls手动标注的数据为待处理的格式
    :param xls_file: 目标文件路径
    :return: 转换后的字典列表
    """
    f = pd.read_excel(xls_file, index=False)
    cnt = 0
    result = []
    while cnt < len(f) - 1:
        if f.text[cnt] == f.text[cnt + 1]:
            temp_dic = {'text': f.text[cnt], 'spo_list': []}
            while cnt < len(f) - 1 and f.text[cnt] == f.text[cnt + 1]:
                temp_dic['spo_list'].append(f.iloc[cnt, 1:].to_dict())
                cnt += 1
            temp_dic['spo_list'].append(f.iloc[cnt, 1:].to_dict())
            cnt += 1
            result.append(temp_dic)
        else:
            temp_dic = {'text': f.text[cnt],
                        'spo_list': [f.iloc[cnt, 1:].to_dict()]}
            result.append(temp_dic)
            cnt += 1

    return result


def split_train_eval_data(origin_file: str, train_file: str,
                          eval_file: str, fraction: float = .2) -> None:
    """
    从原始手动标注数据中分离训练和测试集
    :param origin_file: 原始数据路径
    :param train_file: 保存的训练数据文件路径
    :param eval_file: 保存的测试文件路径
    :param fraction: 分离的测试数据比例，默认总体的20%
    :return:
    """
    origin_data = pd.read_excel(origin_file)
    eval_data = origin_data.sample(frac=fraction)
    train_data = origin_data.drop(index=eval_data.index)
    eval_data.reset_index(drop=True, inplace=True)
    train_data.reset_index(drop=True, inplace=True)
    eval_data.to_excel(eval_file, idnex=False)
    train_data.to_excel(train_file, idnex=False)


def replace(sentence: str, origin: str, replacement: str) -> str:
    """
    替换句子某字符串，此处形式供pandas调用
    :param sentence: 句子
    :param origin: 句子内词语
    :param replacement: 替换词语
    :return: 替换后句子
    """
    return sentence.replace(origin, replacement)


def augment_train_data_with_replacement(train_data: pd.DataFrame,
                                        replace_entity: str,
                                        synonym_func: Union[str, Callable],
                                        fraction: float = .2) -> tuple:
    """
    在pandas数据中对数据进行增强，主要也是同义词替换操作
    :param train_data: 训练数据
    :param replace_entity: 待替换的实体名称
    :param synonym_func: 同义词生成所需的同义词字典或者是生成函数
    :param fraction: 对增强的数据进行测试集采样的比例，默认为0.2
    :return: 增强后的数据，随机分成一定比例的训练集和测试集
    """
    if replace_entity == 'Text':
        need_replace_entity = train_data[train_data.object_type == replace_entity][train_data.predicate == '诉求']
    else:
        need_replace_entity = train_data[train_data.object_type == replace_entity]
    if isinstance(synonym_func, str):
        synonym_list = [s.strip().split(' ')[0] for s in codecs.open(synonym_func) if len(s) < 21]
    else:
        synonym_list = [synonym_func() for _ in range(1000)]

    entity_df = pd.DataFrame()
    entity_sample_list = np.random.choice(synonym_list, size=3000 // len(need_replace_entity))

    for entity in tqdm(entity_sample_list):
        new_s = need_replace_entity.copy()
        target_attr = set(new_s.object) if replace_entity != '案件身份' else set(new_s.subject)
        for origin_entity in target_attr:
            temp_s = new_s[new_s.object == origin_entity] if replace_entity != '案件身份' \
                else new_s[new_s.subject == origin_entity]
            temp_s.text = temp_s.text.apply(replace, args=(origin_entity, entity))
            if replace_entity in ['人物', '案件身份']:
                temp_s.subject = temp_s.subject.apply(replace, args=(origin_entity, entity))
            temp_s.object = temp_s.object.apply(replace, args=(origin_entity, entity))
            entity_df = entity_df.append(temp_s, ignore_index=True)

    eval_sample = entity_df.sample(frac=fraction)
    entity_df_dropped = entity_df.drop(index=eval_sample.index)
    entity_df_dropped.reset_index(inplace=True, drop=True)

    return entity_df_dropped, eval_sample


def generate_random_date(start: tuple = (1910, 1, 1, 0, 0, 0, 0, 0, 0),
                         end: tuple = (2020, 1, 1, 0, 0, 0, 0, 0, 0), ) -> str:
    """
    随机生成日期
    :param start: 随机日期开始日期
    :param end: 随机日期结束日期
    :return: 年月日、年月、年格式的随机日期
    """
    start = time.mktime(start)
    end = time.mktime(end)

    g_date = time.strftime('%Y年%m月%d日', time.localtime(random.randint(start, end)))

    return random.choice([g_date, g_date[:-3], g_date[:5]])


def generate_money_number() -> str:
    """
    随机生成金钱金额
    :return: 数额
    """
    float_number = str(random.randint(100, 10000) / 100)
    han_number = random.choice(list('一二三四五六七八九'))
    unit = random.choice(['千元', '万元', '元'])

    return random.choice([float_number, han_number]) + unit


def generate_random_penalty() -> str:
    """
    随机生成量刑实体
    :return: 随机量刑实体
    """
    penalty_list = ['管制', '拘役', '有期徒刑', '无期徒刑' '死刑']
    random_year = random.choice(
        [random.choice(['', '十']) + random.choice(list('一二三四五六七八九')), random.choice(range(1, 100))])
    random_han_month = list('一二三四五六七八九十') + ['十一', '十二']
    random_month = random.choice([random.choice(random_han_month), random.choice(range(1, 13))])

    return f'{np.random.choice(penalty_list[:3], p=[.2, .2, .6])}{random_year}年{random_month}个月'


synonym_file_list = {
    '罪名': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/罪名字典_699.txt',
    '地点': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/全国地名大全.txt',
    '组织机构': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/机构名词典.txt',
    '人物': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/人名词典.txt',
    '案件身份': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/案件身份.txt',
    '作案方式': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/作案手段.txt',
    '行为': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/坦白行为.txt',
    'Text': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/诉求.txt',
    'TextResult': '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/实体同义词替换词典/诉求结果.txt',
    'Number': generate_money_number,
    '时间': generate_random_date,
    '量刑': generate_random_penalty
}


def convert_xls_to_json(xls_file: str, json_file: str) -> None:
    """
    转换excel文件，处理文件并保存为json文件
    :param xls_file: excel文件
    :param json_file: json文件
    :return:
    """
    temp_result = manage_data(xls_file)
    random.shuffle(temp_result)
    convert_train_valid_data_v2(temp_result, json_file)


def synonym_func(object_type: str) -> str:
    """
    同义词生成函数，从同义词字典随机选取或者通过函数随机生成
    :param object_type: 待生成同义词的实体种类，如罪名，组织机构，时间等
    :return: 某种类的同义词
    """
    value = synonym_file_list[object_type]
    if isinstance(value, str):
        synonym_list = [s.strip().split(' ')[0] for s in codecs.open(value) if len(s) < 21]
        return random.choice(synonym_list)
    else:
        return value()


def augment_train_data_atom(xls_file: str,
                            temp_json_file: str,
                            result_json_file: str,
                            augment_rate: int = 100) -> None:
    """
    对excel文件的样本进行增强，因为excel中的数据为一条数据对应一条spo关系，因此需要转成一条数据对应多条spo（多条spo均属于该样本）；
    之后根据统计的关系数量进行倍数增强，固定倍数参数默认为100，如某关系的数量为20， 则其增强的倍数为 100 // 20 = 5倍
    :param xls_file: excel文件
    :param temp_json_file: excel处理后未转成模型格式数据的文件
    :param result_json_file: 符合模型输入的json文件
    :param augment_rate: 固定增强参数，默认为100
    :return:
    """
    # 从excel一对一返回[{'text':something, 'spo_list': [1,2,3]}]的一对多格式数据
    train_result = manage_data(xls_file)
    train_df = pd.read_excel(xls_file, index=False)
    predicate_count_dict = dict(train_df.predicate.value_counts())
    aug_train_result = []

    for example in tqdm(train_result):
        example_copy = deepcopy(example)
        text, spo_list = example_copy['text'], example_copy['spo_list']
        for spo in spo_list:
            for _ in range(augment_rate // predicate_count_dict[spo['predicate']]):
                new_example = {}
                new_spo = deepcopy(spo)
                spo_list_copy = deepcopy(spo_list)
                replacer = synonym_func(spo['object_type'])
                new_spo['object'] = replacer
                other_spo = [os for os in spo_list_copy if os != spo]
                for i, os in enumerate(other_spo):
                    if os['object'] == spo['object']:
                        other_spo[i]['object'] = replacer
                    if os['subject'] == spo['object']:
                        other_spo[i]['subject'] = replacer

                if spo['object'] in spo['subject']:
                    temp_subject = spo['subject']
                    temp_text = text.replace(temp_subject, f'[origin_subject_token]')
                    temp_text = temp_text.replace(spo['object'], replacer)
                    temp_text = temp_text.replace(f'[origin_subject_token]', temp_subject)
                    new_example['text'] = temp_text
                else:
                    new_example['text'] = text.replace(spo['object'], replacer)

                new_example['spo_list'] = other_spo + [new_spo]
                aug_train_result.append(new_example)

    random.shuffle(aug_train_result)
    print(len(aug_train_result))
    save_result_to_json(temp_json_file, aug_train_result)
    convert_train_valid_data_v2(aug_train_result, result_json_file)


if __name__ == '__main__':
    # train_file = '../data_path/datas/manually_labelled_train.xls'
    # train_json = '../data_path/datas/manually_labelled_train_large.json'
    # temp_train_json = '../data_path/datas/train_text_and_spo_list_large.json'
    # augment_train_data_atom(train_file, temp_train_json, train_json, augment_rate=3000)
    # convert_xls_to_json(train_file, train_json)

    # eval_file = '../data_path/datas/manually_labelled_eval.xls'
    # eval_json = '../data_path/data/datasets/wenshu/manually_labelled_eval_augmented.json'
    # temp_eval_json = '../data_path/data/datasets/wenshu/text_and_spo_list_eval.json'
    # # convert_xls_to_json(eval_file, eval_json)
    # augment_train_data_atom(eval_file, temp_eval_json, eval_json, augment_rate=50)

    # eval_file = '../data_path/datas/test_single_eval.xls'
    # eval_json = '../data_path/datas/test_single_eval.json'
    # temp_eval_json = '../data_path/datas/test_single_eval_temp.json'
    # # convert_xls_to_json(eval_file, eval_json)
    # augment_train_data_atom(eval_file, temp_eval_json, eval_json, augment_rate=20)

    convert_data_types('../data_path/datas/all_schema_combined',
                       '../data_path/datas/wenshu_types_combined.json')

    # baidu额外数据
    # extra_baidu_eval_path = '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/extra_baidu_eval.json'
    # extra_baidu_eval_final_path = '/home/ljh/Projects/ee_dl_ie/pst/tripples_extraction/data_path/datas/extra_baidu_eval_final.json'
    # extra_baidu_eval = json.load(codecs.open(extra_baidu_eval_path))
    # convert_train_valid_data_v2(extra_baidu_eval, extra_baidu_eval_final_path)
