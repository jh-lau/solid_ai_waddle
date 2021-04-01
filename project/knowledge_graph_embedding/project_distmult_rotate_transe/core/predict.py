"""
  @Author       : liujianhan
  @Date         : 2018/5/17 下午5:10
  @Project      : KGE
  @FileName     : predict.py
  @Description  : 预测模块
"""
import os
from typing import Tuple, Dict

import numpy as np
import torch

from .train import read_triple


class EmbeddingPrediction:
    def __init__(self, head: np.ndarray,
                 tail: np.ndarray,
                 relation: np.ndarray,
                 entity_emb: np.ndarray,
                 relation_emb: np.ndarray,
                 id2relation: Dict,
                 id2entity: Dict):
        self.head = torch.tensor(head)
        self.tail = torch.tensor(tail)
        self.relation = torch.tensor(relation)
        self.entity_emb = torch.tensor(entity_emb)
        self.relation_emb = torch.tensor(relation_emb)
        self.id2relation = id2relation
        self.id2entity = id2entity

    def __call__(self, model_name) -> Dict:
        model_func = {
            'TransE': self.TransE_emb,
            'DistMult': self.DistMult_emb,
            'ComplEx': self.ComplEx_emb,
            'RotatE': self.RotatE_emb,
        }
        try:
            return model_func[model_name]()
        except KeyError:
            return {"预测结果": f"不支持的模型名称{model_name}, 请确保模型名称为{list(model_func.keys())}之一。"}
        except Exception as e:
            print(e)

    def TransE_emb(self) -> Dict:
        """不支持对称关系"""
        head_pred = (self.entity_emb - (self.tail - self.relation)).norm(p=1, dim=1)
        tail_pred = (self.entity_emb - (self.head + self.relation)).norm(p=1, dim=1)
        relation_pred = (self.relation_emb - (self.head + self.tail)).norm(p=1, dim=1)

        return self.get_sorted_rank(head_pred, relation_pred, tail_pred)

    def DistMult_emb(self):
        """只支持对称关系"""
        # todo 公式结果跟service.py中的结果不同
        head_pred = (self.entity_emb / (self.relation * self.tail)).sum(dim=1)
        tail_pred = (self.entity_emb / self.head * self.relation).sum(dim=1)
        relation_pred = (self.relation_emb / self.tail / self.head).sum(dim=1)

        return self.get_sorted_rank(head_pred, relation_pred, tail_pred)

    def ComplEx_emb(self):
        """不支持组合关系"""
        # todo 公式结果跟service.py中的结果不同
        re_head, im_head = torch.chunk(self.head, 2, dim=1)
        re_relation, im_relation = torch.chunk(self.relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.tail, 2, dim=1)

        re_entity_emb, im_entity_emb = torch.chunk(self.entity_emb, 2, dim=1)
        re_relation_emb, im_relation_emb = torch.chunk(self.relation_emb, 2, dim=1)

        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_head_pred = re_entity_emb / re_score
        im_head_pred = re_entity_emb / im_score
        head_pred = torch.cat([re_head_pred, im_head_pred], dim=-1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_tail_pred = re_entity_emb / re_score
        im_tail_pred = re_entity_emb / im_score
        tail_pred = torch.cat([re_tail_pred, im_tail_pred], dim=-1)

        return self.get_sorted_rank(head_pred, self.relation_emb, tail_pred)

    def RotatE_emb(self):
        # todo 公式结果跟service.py中的结果不同
        re_entity_emb, _ = torch.chunk(self.entity_emb, 2, dim=-1)
        re_head, _ = torch.chunk(self.head, 2, dim=-1)
        re_tail, _ = torch.chunk(self.tail, 2, dim=-1)
        head_pred = ((re_entity_emb - re_tail) / self.relation).norm(1, dim=1)
        relation_pred = ((self.relation_emb - re_tail) / re_head).norm(1, dim=1)
        tail_pred = (re_entity_emb - (re_head * self.relation)).norm(1, dim=1)

        return self.get_sorted_rank(head_pred, relation_pred, tail_pred)

    def get_sorted_rank(self,
                        pred_head: torch.tensor,
                        pred_relation: torch.tensor,
                        pred_tail: torch.tensor,
                        top_k: int = 10) -> Dict:
        head_rank = torch.argsort(pred_head).tolist()[:top_k]
        tail_rank = torch.argsort(pred_tail).tolist()[:top_k]
        relation_rank = torch.argsort(pred_relation).tolist()[:top_k]
        try:
            head_res = [self.id2entity[str(r)] for r in head_rank]
            tail_res = [self.id2entity[str(r)] for r in tail_rank]
            relation_res = [self.id2relation[str(r)] for r in relation_rank]
            return {"头实体预测": head_res, "关系预测": relation_res, "尾实体预测": tail_res}
        except KeyError:
            return {"预测结果": f"键值不存在，请确认实体或者关系的向量维度是否正确。"}


def get_entity_relation_with_id(data_path: str, separator: str = '\t') -> Tuple:
    """
    加载实体和关系字典文件与所有正确三元组信息
    @param data_path: 文件路径
    @param separator: 文件中每行内容的分割符
    @return: 实体、关系与id映射字典
    """
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        entity2id, id2entity = dict(), dict()
        for line in fin:
            eid, entity = line.strip().split(separator)
            entity2id[entity] = int(eid)
            id2entity[eid] = entity

    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id, id2relation = dict(), dict()
        for line in fin:
            rid, relation = line.strip().split(separator)
            relation2id[relation] = int(rid)
            id2relation[rid] = relation

    train_triples = read_triple(os.path.join(data_path, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_path, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_path, 'test.txt'), entity2id, relation2id)
    all_true_triples = train_triples + valid_triples + test_triples

    return entity2id, id2entity, relation2id, id2relation, all_true_triples

