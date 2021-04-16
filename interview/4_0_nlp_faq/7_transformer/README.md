1. Transformer和RNN相比有哪些区别
    - 优势
        - 可以并行计算（位置编码的必要性）
        - 防止RNN存在的梯度消失的问题
        - 可解释性较好：不同单词之间的相关性（注意力）有多大
    - 劣势
        - RNN可以轻松解决复制输入的任务
        - 实际碰到的序列长度比训练时长，Transformer无法处理
        - RNN图灵完备（近似任意Turing计算机可以解决的算法），Transformer不是
    
2. Transformer的复杂度，如何降低复杂度
    - holder
    
3. Transformer实践注意事项
    - 多头注意力的个数必须能被嵌入维度整除，比如：(512 / 8)
    - LabelSmoothing
    - Noam Learning Rate Schedule：让学习率线性增长到某个最大的值，然后再按指数的方式衰减
    
100. holder


# Filtered References
> [1.The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
>
> [2.再回首 DeepLearning 遇见了 Transformer](https://mp.weixin.qq.com/s/a3sbbCYioAPkK471BRBPyw)
>   - 配图中错误：注意力计算中，根号`dk`不是表示`head`个数，而是矩阵`K`的维度
