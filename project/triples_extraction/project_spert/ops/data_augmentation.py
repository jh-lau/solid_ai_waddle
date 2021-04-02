"""
  @Author       : liujianhan
  @Date         : 2018/3/18 下午2:38
  @Project      : triples_extraction
  @FileName     : data_augmentation.py
  @Description  : 数据增强模块
"""
import codecs
import random
import re
from glob import glob
from typing import List

import jieba
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from synonyms import synonyms
from tqdm import tqdm
import pandas as pd

stop_words = set([s.rstrip() for s in codecs.open('/home/ljh/Projects/ee_dl_ie/法律专有词典/stopwords.txt')])


def model_train(file_path: str, model_name: str) -> Word2Vec:
    """
    训练词向量
    :param file_path: 清洗过的语料
    :param model_name: 保存的模型名称
    :return: 模型文件
    """
    open_file = codecs.open(file_path, 'r', encoding='utf-8')
    sentences = LineSentence(open_file)
    model = Word2Vec(sentences, size=100, window=10, min_count=20, workers=4)
    model.save(model_name)
    return model


class DataAugmentation:
    def __init__(self, sentence):
        self.sentence = sentence

    @classmethod
    def get_synonyms(cls, word: str, choice: bool = True) -> List[str]:
        """
        获取词汇的同义词列表
        :param word: 目标词汇
        :param choice: 选择哪一种Word2Vec同义词向量，真为自训练，否则为第三方库
        :return: 同义词列表
        """
        if choice:
            model = Word2Vec.load('/home/ljh/Projects/ee_dl_ie/triples_extraction/data_path/data/datasets/corpus/wenshu_word2vec')
            try:
                return [s[0] for s in model.wv.most_similar(word)]
            except KeyError:
                return []

        return synonyms.nearby(word)[0]

    def synonym_replacement(self) -> str:
        """
        对一个句子进行同义词替换
        :return: 同义词替换后的句子
        """
        word_list = jieba.lcut(self.sentence)
        result = ''
        for index, word in enumerate(word_list):
            syn = self.get_synonyms(word)
            if syn:
                result += random.choice(syn[1:])
            else:
                result += word
        return result

    @classmethod
    def split_sentence(cls, sentences: str) -> list:
        """
        文档句子分句
        :param sentences: 目标句子
        :return: 分句列表
        """
        if isinstance(sentences, str):
            return [s for s in re.split(r'[。！？\n]', sentences) if s]

    @staticmethod
    def remove_stopwords(sentence: str) -> list:
        """
        去除句子中的停用词
        :param sentence: 目标句子
        :return: 去除停用词后的词汇列表
        """
        if sentence:
            word_list = jieba.lcut(sentence)
            return [word for word in word_list if word not in stop_words]

    @classmethod
    def process_pipeline(cls, sentences: str, file_path: str) -> None:
        """
        分句并去除停用词后保存到文件
        :param sentences: 文档字符串内容
        :param file_path: 保存文件路径
        :return:
        """
        sentence_list = cls.split_sentence(sentences)
        if sentence_list:
            with open(file_path, 'a', encoding='utf-8') as file:
                for sentence in sentence_list:
                    clean_list = cls.remove_stopwords(sentence)
                    file.write(' '.join(clean_list) + '\n')


if __name__ == '__main__':
    file_list = glob('data_path/data/datasets/wenshu/刑事案件数据_*.xls')
    for file in tqdm(file_list):
        print(file)
        read_file = pd.read_excel(file, index_col=0)
        keys = ['本院查明', '本院认为', '裁判结果']
        for key in keys:
            read_file.loc[:, key].apply(DataAugmentation.process_pipeline, args='clean_wenshu_corpus.txt')
    s = '显然，马晓梅的行为构成了很严重的集资诈骗罪'
    print(DataAugmentation(s).synonym_replacement())
