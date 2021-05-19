1. GraphSAGE（SAmple and aggreGatE）
    - 归纳式学习算法（inductive），解决了未知节点无法`Embedding`的问题
    - 将 GCN 扩展到无监督的归纳学习任务中，还泛化了 GCN 的聚合函数
    - 步骤
        - 首先对节点的一阶和二阶邻居节点进行采样
        - 然后根据聚合函数聚合邻居节点的特征
        - 最后得到节点的`Embedding`向量
1. PageRank
    - 原理
    - SIF调优过程
1. GCN，GNN，GAT