1. CRF vs HMM
    - 无向图 vs 有向图
    - 判别式 vs 生成式
    - 无标注偏置（全局归一化，枚举了整个隐状态序列的全部可能，消除局部归一化带来的标注偏置问题） vs 有标注偏置（局部归一化的原因，隐状态会倾向转移到后续状态可能更少的状态上，以提高整体的后验概率，导致标注偏置问题）
    
1. CRF损失函数定义
    - 目标标签序列概率/所有标签组合概率
    - 目标标签概率：发射概率 + 转移概率
    - 模型训练完成后，推断过程用到维特比进行解码



### References
1. [概率图模型体系：HMM、MEMM、CRF](https://zhuanlan.zhihu.com/p/33397147)
2. [CRF-Layer-on-the-Top-of-BiLSTM](https://createmomo.github.io/2017/11/11/CRF-Layer-on-the-Top-of-BiLSTM-5/)
