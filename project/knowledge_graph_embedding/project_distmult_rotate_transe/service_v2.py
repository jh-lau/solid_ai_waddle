"""
  @Author       : liujianhan
  @Date         : 2020/5/21 下午7:20
  @Project      : KGE
  @FileName     : service_v2.py
  @Description  : 根据图向量直接进行运算的推理
"""
import time

from .core.predict import *

entity_emb, relation_emb, entity2id, id2entity, relation2id, id2relation \
    = None, None, None, None, None, None


def load_model(model_path: str,
               data_path: str) -> None:
    """
    加载函数
    @param model_path: 模型文件夹路径
    @param data_path: 数据文件夹路径
    @return:
    """
    global entity_emb, relation_emb, entity2id, id2entity, relation2id, id2relation
    entity_emb = np.load(os.path.join(model_path, 'entity_embedding.npy'))
    relation_emb = np.load(os.path.join(model_path, 'relation_embedding.npy'))
    entity2id, id2entity, relation2id, id2relation, _ = \
        get_entity_relation_with_id(data_path)


def inference(target_triple: str, model_name: str = 'TransE') -> Dict:
    """
    推理函数
    @param target_triple: 目标正确三元组，格式如'摩耶号/Maya巡洋舰 建造时间 1928年'
    @param model_name: 使用的模型名称，支持[TransE,RotatE,ComplEx,DistMult]
    @return: 推理结果
    """
    try:
        target_triple = target_triple.split()
        head = entity_emb[entity2id[target_triple[0]], :].reshape(1, -1)
        tail = entity_emb[entity2id[target_triple[2]], :].reshape(1, -1)
        relation = relation_emb[relation2id[target_triple[1]], :].reshape(1, -1)
    except KeyError as e:
        return {'预测结果': f'实体或者关系 <{e}> 不存在，请确保输入的实体或者关系已存在。'}

    prediction = EmbeddingPrediction(head, tail, relation, entity_emb,
                                     relation_emb, id2relation, id2entity)
    return prediction(model_name)


if __name__ == '__main__':
    t1 = time.time()
    load_model('data_path/model/TransE_cn_military_300k_13/',
               'data_path/data/cn_military_300k')

    t2 = time.time()
    print(f"加载耗时： {t2 - t1: .3f}s")
    print('预测头实体：', inference('摩耶号/Maya巡洋舰 建造时间 1928年'))
    print(f"推理耗时： {time.time() - t2: .3f}s")
    """
    >>> 加载耗时：  0.260s
    >>> 预测头实体： {'头实体预测': ['吹雪级/Fubuki驱逐舰', '“阿尔·希蒂克”(Alsiddiq)级快速导弹巡逻舰', 'TNC45级通用快速巡逻舰', 
    '“斯德哥摩尔”（Stockholm）级轻型护卫舰', '“西巴劳”(Sibarau)级通用巡逻舰', '西约特兰级/射手级/南曼兰级', '晓级/Akatsuki驱逐舰', 
    '“海蛇”(Sjoormen)级(A12)常规攻击型潜艇', '“阿米代尔”(Armidale)级巡逻舰', '“三伙伴”(Tripartite)级猎雷艇'], 
    '关系预测': ['舰舰导弹', '研发时间', '诞生时间', '生产年限', '研制时间', '名称', '首次轨道发射', '退役时间', '服役时间', '首飞时间'], 
    '尾实体预测': ['1930年', '1932年', '1929年', '1927年4月28日', '1927年', '1941年', '1924年', '1935年', '1937年', '1933年']}
    >>> 推理耗时：  0.047s
    """
