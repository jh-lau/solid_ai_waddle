"""
  @Author       : liujianhan
  @Date         : 2018/2/26 下午3:42
  @Project      : triples_extraction
  @FileName     : utils.py
  @Description  : 数据处理公用函数
"""
import codecs
import hashlib
import http.client
import json
import random
import urllib
from typing import Union, Dict, List

import yaml


def translate_baidu(query, src: str = 'auto', dest: str = 'zh') -> List[Dict]:
    """
    百度翻译api调用
    :param query: 待翻译句子
    :param src: 原始语言
    :param dest: 目标语言
    :return: 翻译的结果，为字典列表格式：[{'src': query, 'dst': translation}]
    """
    app_id = '20200324000403795'
    secret_key = 'ED2NkcPk0M8FVVl11Tje'

    http_client = None
    head_url = '/api/trans/vip/translate'

    salt = random.randint(32768, 65536)

    sign = app_id + query + str(salt) + secret_key
    sign = hashlib.md5(sign.encode()).hexdigest()
    target_url = f'{head_url}?appid={app_id}&q={urllib.parse.quote(query)}&from={src}&to={dest}' \
                 f'&salt={str(salt)}&sign={sign}'

    try:
        http_client = http.client.HTTPConnection('api.fanyi.baidu.com')
        http_client.request('GET', target_url)

        response = http_client.getresponse()
        result_all = response.read().decode("utf-8")
        result = json.loads(result_all)

        return result['trans_result']
    except Exception as e:
        pass
    finally:
        if http_client:
            http_client.close()


def substitute_with_back_translate_baidu(query: str, dst: str = 'en') -> str:
    """
    通过百度翻译进行反向翻译
    :param query: 待翻译句子
    :param dst: 目标语言
    :return: 反向翻译过的句子
    """
    try:
        temp_trans = translate_baidu(query, dest=dst)[0]['dst']
        new_sentence = translate_baidu(temp_trans, dest='zh')[0]['dst']

        return new_sentence
    except:
        pass


def substitute_with_back_translate_google(query: str, src: str = 'zh-cn',
                                          dst: str = 'en') -> str:
    """
    通过谷歌翻译进行反向翻译
    :param query: 待翻译句子
    :param src: 句子源语言
    :param dst: 目标语言
    :return: 反向翻译后的句子
    """
    from googletrans import Translator as Translator
    try:
        translator = Translator(service_urls=['translate.google.cn'])
        temp_trans = translator.translate(query, src=src, dest=dst).text
        new_sentence = translator.translate(temp_trans, src=dst, dest=src).text

        return new_sentence
    except:
        pass


grammar_tree = {
    'simple_sentence': [['subject*', 'predicate*', 'object*'], ['subject*', 'predicate*']],
    'subject*': [['attribute*', 'subject']],
    'attribute*': [['attribute'], ['null']],
    'predicate*': [['adverbial*', 'predicate', 'complement*']],
    'adverbial*': [['adverbial'], ['null']],
    'complement*': [['complement'], ['null']],
    'object*': [['attribute*', 'object']],
    'subject': [['李大锤'], ['刘德华'], ['梅艳芳']],
    'predicate': [['涉嫌'], ['不构成'], ['诈骗']],
    'object': [['行为主义'], ['集资诈骗罪'], ['自首行为']],
    'attribute': [['令人发指的'], ['漂亮的'], ['聪明的']],
    'adverbial': [['从容地'], ['狡猾地'], ['阴险地']],
    'complement': [['极了'], ['死了'], ['得很'], ['好了']]
}


def generate_sentence_from_grammar_tree(gram_tree: Dict, target: str) -> str:
    """
    根据自定义的语法模板进行过句子生成
    :param gram_tree: 自定义语法模板
    :param target: 目标句式结构，如simple_sentence
    :return: 根据随机模板生成的句子
    """
    if target not in gram_tree:
        return target
    expanded = [generate_sentence_from_grammar_tree(gram_tree, s)
                for s in random.choice(gram_tree[target])]

    return ''.join([e for e in expanded if e != 'null'])


def encode_list(target_list: list) -> list:
    """
    序列编码，为了后续取索引，用于句子英文单词和数字不拆分的情况
    :param target_list: 待编码序列
    :return:
    """
    result = []
    if not target_list:
        return result

    for x in target_list:
        if len(x) > 1:
            result.append('#')
        else:
            result.append(x)

    return result


def get_list_index(target_list: list, sub_str: str) -> tuple:
    """
    获取字序列在序列中的索引，用于句子英文单词和数字不拆分的情况
    :param target_list: 目标序列
    :param sub_str: 子序列
    :return: 索引元组
    """
    if not target_list or not sub_str:
        return -1, -1
    encode_target = ''.join(encode_list(target_list))
    encode_sub = ''.join(encode_list(split_chi_char(sub_str)))
    start_index = encode_target.index(encode_sub)
    end_index = start_index + len(encode_sub)

    return start_index, end_index


def get_list_index_v2(target_list: list, sub_list: list) -> tuple:
    """
    获取子列表在目标列表中的索引
    a1 = ['a', 'abcd', 'e', 'ab', 'abcd', 'e', 'f', 'a', 'e', 'd', 's', 'abcdd', 'e']
    a2 = ['abcd', 'e', 'f', 'a', 'e', 'd', 's']
    b = ['abcd', 'e', 'f']
    get_list_index_v2(a1, b) --> 4, 6
    get_list_index_v2(a2, b) --> 0, 2
    :param target_list: 目标列表
    :param sub_list: 子列表
    :return: 子列表在目标列表中的起讫索引
    """
    if not target_list or not sub_list:
        return -1, -1

    result = []
    i, j = 0, 0
    while i < len(target_list):
        while j < len(sub_list):
            if target_list[i] == sub_list[j]:
                result.append(i)
                i += 1
                j += 1
            else:
                result = []
                j = 0
                break
        i += 1

    return result[0], result[-1]


def split_chi_char(string: str) -> list:
    """
    切分中文句子为字符，保留英文词汇与数字
    :param string: 目标句子
    :return: 切割结果
    """
    result = []
    if not string:
        return result
    flag = False
    for s in string:
        if s.lower().islower() or s.isdigit():
            if flag:
                result[-1] += s
            else:
                result.append(s)
                flag = True
        elif s == ' ':
            flag = False
            continue
        else:
            flag = False
            result.append(s)
    return result


def convert_data_types(file_path: str, save_path: str) -> None:
    """
    转换语料实体类型与关系类型为目标格式并保存
    :param file_path: 待转换文件
    :param save_path: 保存文件路径
    :return:
    """
    final_type = {}
    entities_type, relations_type = {}, {}
    for line in codecs.open(file_path):
        if line.strip():
            line = eval(line)
            object_entity = line['object_type']
            subject_entity = line['subject_type']
            relation = line['predicate']

            entities_type[object_entity] = {"short": object_entity,
                                            "verbose": object_entity}

            entities_type[subject_entity] = {"short": subject_entity,
                                             "verbose": subject_entity}
            relations_type[relation] = {"short": relation,
                                        "verbose": relation,
                                        "symmetric": False}
    final_type["entities"] = entities_type
    final_type['relations'] = relations_type

    save_result_to_json(save_path, final_type)


def convert_train_valid_data(file_path: str, save_path: str, keep_alpha_and_digit: bool = True) -> None:
    """
    转换语料为目标格式
    :param file_path: 语料路径
    :param save_path: 保存路径
    :param keep_alpha_and_digit: 英文单词（John）和数字（1988）实体是否保持而非拆分为单字符，默认保持
    :return:
    """
    result = []
    for line_num, line in enumerate(codecs.open(file_path, 'r', 'utf-8')):
        try:
            line = eval(line)
            spo_list = line['spo_list']

            doc_dict = {'tokens': [],
                        'entities': [],
                        'relations': []}

            if keep_alpha_and_digit:
                doc_dict['tokens'] = split_chi_char((line['text']).lower())
            else:
                doc_dict['tokens'] = list(line['text'].lower())

            for spo_dict in spo_list:
                ob_type = spo_dict['object_type']
                sb_type = spo_dict['subject_type']
                if keep_alpha_and_digit:
                    ob_start, ob_end = get_list_index(doc_dict['tokens'], spo_dict['object'].lower())
                    sb_start, sb_end = get_list_index(doc_dict['tokens'], spo_dict['subject'].lower())
                else:
                    ob_start = line['text'].lower().index(spo_dict['object'].lower())
                    ob_end = ob_start + len(spo_dict['object'].lower())
                    sb_start = line['text'].lower().index(spo_dict['subject'].lower())
                    sb_end = sb_start + len(spo_dict['subject'].lower())

                doc_dict["entities"].append({"type": ob_type,
                                             "start": ob_start,
                                             "end": ob_end})
                tail_index = len(doc_dict['entities']) - 1

                doc_dict["entities"].append({"type": sb_type,
                                             "start": sb_start,
                                             "end": sb_end})
                head_index = len(doc_dict['entities']) - 1

                doc_dict['relations'].append({"type": spo_dict['predicate'],
                                              'head': head_index,
                                              'tail': tail_index})
            doc_dict['orig_id'] = line_num

            if doc_dict['relations']:
                result.append(doc_dict)
        except Exception as e:
            print(e)
            print(line['text'], line['spo_list'])
            print()

    save_result_to_json(save_path, result)


def convert_train_valid_data_v2(target_list: list, save_path: str, keep_alpha_and_digit: bool = True) -> None:
    """
    转换语料为目标格式
    :param target_list: 目标字典列表
    :param save_path: 保存路径
    :param keep_alpha_and_digit: 英文单词（John）和数字（1988）实体是否保持而非拆分为单字符，默认保持
    :return:
    """
    result = []
    if not target_list:
        pass
    for line_num, line in enumerate(target_list):
        try:
            spo_list = line['spo_list']

            doc_dict = {'tokens': [],
                        'entities': [],
                        'relations': []}

            if keep_alpha_and_digit:
                doc_dict['tokens'] = split_chi_char((line['text']).lower())
            else:
                doc_dict['tokens'] = list(line['text'].lower())

            for spo_dict in spo_list:
                ob_type = spo_dict['object_type']
                sb_type = spo_dict['subject_type']
                if keep_alpha_and_digit:
                    ob_start, ob_end = get_list_index(doc_dict['tokens'], spo_dict['object'].lower())
                    sb_start, sb_end = get_list_index(doc_dict['tokens'], spo_dict['subject'].lower())
                else:
                    ob_start = line['text'].lower().index(spo_dict['object'].lower())
                    ob_end = ob_start + len(spo_dict['object'].lower())
                    sb_start = line['text'].lower().index(spo_dict['subject'].lower())
                    sb_end = sb_start + len(spo_dict['subject'].lower())

                doc_dict["entities"].append({"type": ob_type,
                                             "start": ob_start,
                                             "end": ob_end})
                tail_index = len(doc_dict['entities']) - 1

                doc_dict["entities"].append({"type": sb_type,
                                             "start": sb_start,
                                             "end": sb_end})
                head_index = len(doc_dict['entities']) - 1

                doc_dict['relations'].append({"type": spo_dict['predicate'],
                                              'head': head_index,
                                              'tail': tail_index})
            doc_dict['orig_id'] = line_num

            if doc_dict['relations']:
                result.append(doc_dict)
        except Exception as e:
            print(e)
            print(line['text'], line['spo_list'])
            print()

    save_result_to_json(save_path, result)


def save_result_to_json(file_name: str, data: Union[list, dict], ensure_ascii: bool = False) -> None:
    """
    保存文件的为json格式结果
    :param file_name: 文件名
    :param data: 待保存数据
    :param ensure_ascii: 保存文件是否正常显示可阅读
    :return:
    """
    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii)
    except FileNotFoundError:
        pass


def parse_yaml_parameters(yaml_file: str) -> dict:
    """
    读取yaml参数配置
    :param yaml_file: 参数配置文件
    :return: 参数字典
    """
    file = codecs.open(yaml_file, 'r', encoding='utf-8')
    file_data = yaml.load(file, Loader=yaml.FullLoader)
    return file_data


if __name__ == '__main__':
    test_case = ['被告人王增敏以诈骗罪于2020年3月13日被石家庄市新华区人民法院判处有期徒刑十三年，并判处罚金十万元。',
                 '经审理查明：2013年11月初，上诉人罗光通过与酒鬼酒供销公司总经理郝某等工作人员多次洽谈后，以金某公司的名义与酒鬼酒供'
                 '销公司签订“异地存款销酒”合作协议，但是由于罗光没有找到贴息方而未果。11月底，罗光又与郝某联系，郝某安排方某等人与罗光就存款销酒合作事宜再次进行洽谈。']
    for tc in test_case:
        print(tc)
        print('google translate:', substitute_with_back_translate_google(tc))
        print('baidu translate:', substitute_with_back_translate_baidu(tc))
