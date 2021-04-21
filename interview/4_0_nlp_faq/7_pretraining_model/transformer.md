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
    - todo
    
3. Transformer实践注意事项
    - 多头注意力的个数必须能被嵌入维度整除，比如：(512 / 8)
    - LabelSmoothing
    - Noam Learning Rate Schedule：让学习率线性增长到某个最大的值，然后再按指数的方式衰减

4. self-attention vs attention
    - todo

5. Transformer的点积模型做缩放的原因是什么
    - 极大的点积值将整个 softmax 推向梯度平缓区，使得收敛困难
    - 将方差控制为1，也就有效地控制了前面提到的梯度消失的问题
    - 引申：为什么在其他 softmax 的应用场景，不需要做 scaled
    
6. 关于 mask 机制，在 Transformer 中有 attention、encoder 和 decoder 中不同的应用区别是什么
    - todo
    
7. transformer如何捕获序列中的顺序信息
    - 通过使用 Transformer 我们可以得到一个对于输入信息的 embedding vector，要给模型增加捕获序列顺序的能力，我们创建一个和输入序列等长的新序列，这个序列里包含序列中的顺序信息，我们把这个序列和原有序列进行相加，从而得到输入到 Transformer 的序列。那应该怎样表示序列中的位置信息呢？
        - position embeddings 对每一个位置生成一个向量，然后我们使用模型的学习能力来学习到这些位置的 vector
        - position encodings 选择一个 function 来生成每个位置的 vector，并且让模型网络去找出该如何去理解这些 encoding vector。这样做的好处是，对于一个选择的比较好的function，网络模型能够处理那些在训练阶段没有见过的序列位置 vector（虽然这也并不是说这些没见过的位置 vector 一定能够表现的很好，但是好在是我们可以有比较直接的方法来测试他们）
        
8. Transformer的Encoder端和Decoder端是如何进行交互的？和一般的seq2seq有什么差别
    - todo
    
9. 不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵，会有什么问题
    
100. todo

# Filtered References
- [1.The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [2.再回首 DeepLearning 遇见了 Transformer](https://mp.weixin.qq.com/s/a3sbbCYioAPkK471BRBPyw)
    - 配图中错误：注意力计算中，根号`dk`不是表示`head`个数，而是矩阵`K`的维度
- [超细节的BERT/Transformer知识点](https://zhuanlan.zhihu.com/p/132554155)
