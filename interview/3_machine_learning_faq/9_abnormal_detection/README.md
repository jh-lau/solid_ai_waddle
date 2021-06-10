1. 孤立森林的原理
	- 理论基础
	    - 异常数据占比小
	    - 异常数据特征值与正常点的差异很大
    - 创新点
        - Partial models：在训练过程中，每棵孤立树都是随机选取部分样本；
        - No distance or density measures：不同于 KMeans、DBSCAN 等算法，孤立森林不需要计算有关距离、密度的指标，可大幅度提升速度，减小系统开销；
        - Linear time complexity：因为基于 ensemble，所以有线性时间复杂度。通常树的数量越多，算法越稳定；
        - Handle extremely large data size：由于每棵树都是独立生成的，因此可部署在大规模分布式系统上来加速运算。
    - 局限
        - 需要设定异常值占比，占比越接近实际值，效果越好


### Filtered Reference
1. [Github:Anomaly Detection Learning Resources](https://github.com/yzhao062/anomaly-detection-resources)