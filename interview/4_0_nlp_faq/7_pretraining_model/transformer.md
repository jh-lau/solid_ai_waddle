1. Transformer和RNN相比有哪些区别？
    - 优势
        - 可以并行计算（位置编码的必要性）
        - 防止RNN存在的梯度消失的问题
        - 可解释性较好：不同单词之间的相关性（注意力）有多大
    - 劣势
        - RNN可以轻松解决复制输入的任务
        - 实际碰到的序列长度比训练时长，Transformer无法处理
        - RNN图灵完备（近似任意Turing计算机可以解决的算法），Transformer不是

2. Transformer实践注意事项？
    - 多头注意力的个数必须能被嵌入维度整除，比如：(512 / 8)
    - LabelSmoothing
    - Noam Learning Rate Schedule：让学习率线性增长到某个最大的值，然后再按指数的方式衰减

3. Transformer的点积模型做缩放的原因是什么？
    - 极大的点积值将整个`softmax`推向梯度平缓区，使得收敛困难
    - 将方差控制为1，也就有效地控制了前面提到的梯度消失的问题
    - 引申：为什么在其他 softmax 的应用场景，不需要做 scaled
        - 这个时候的梯度形式改变，不会出现极大值导致梯度消失的情况了
    
4. transformer如何捕获序列中的顺序信息？
    - 通过使用 Transformer 我们可以得到一个对于输入信息的 embedding vector，要给模型增加捕获序列顺序的能力，我们创建一个和输入序列等长的新序列，这个序列里包含序列中的顺序信息，我们把这个序列和原有序列进行相加，从而得到输入到 Transformer 的序列。
    - 怎样表示序列中的位置信息
        - position embeddings 对每一个位置生成一个向量，然后我们使用模型的学习能力来学习到这些位置的 vector
        - position encodings 选择一个 function 来生成每个位置的 vector，并且让模型网络去找出该如何去理解这些 encoding vector。这样做的好处是，对于一个选择的比较好的function，网络模型能够处理那些在训练阶段没有见过的序列位置 vector（虽然这也并不是说这些没见过的位置 vector 一定能够表现的很好，但是好在是我们可以有比较直接的方法来测试他们）

5.Transformer为何使用多头注意力机制？（为什么不使用一个头）？
    - 多头可以使参数矩阵形成多个子空间，矩阵整体的size不变，只是改变了每个head对应的维度大小，这样做使矩阵对多方面信息进行学习，但是计算量和单个head差不多
    
6.Transformer计算attention的时候为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？
    - K和Q的点乘是为了得到一个attention score 矩阵，用来对V进行提纯。K和Q使用了不同的W_k, W_Q来计算，可以理解为是在不同空间上的投影。正因为 有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高

7.在计算attention score的时候如何对padding做mask操作？
    - 对需要mask的位置设为负无穷，再对attention score进行相加

8.为什么在进行多头注意力的时候需要对每个head进行降维？
    - 将原有的高维空间转化为多个低维空间并再最后进行拼接，形成同样维度的输出，借此丰富特性信息，降低了计算量

9.还有哪些关于位置编码的技术，各自的优缺点是什么？
    - 相对位置编码（RPE）
        - 1.在计算attention score和weighted value时各加入一个可训练的表示相对位置的参数
        - 2.在生成多头注意力时，把对key来说将绝对位置转换为相对query的位置
        - 3.复数域函数，已知一个词在某个位置的词向量表示，可以计算出它在任何位置的词向量表示。前两个方法是词向量+位置编码，属于亡羊补牢，复数域是生成词向量的时候即生成对应的位置信息

10.Encoder端和Decoder端是如何进行交互的？
    - 通过转置encoder_ouput的seq_len维与depth维，进行矩阵两次乘法，即q*kT*v输出即可得到target_len维度的输出

11.Decoder阶段的多头自注意力和encoder的多头自注意力有什么区别？
    - Decoder有两层mha，encoder有一层mha，Decoder的第二层mha是为了转化输入与输出句长，Decoder的请求q与键k和数值v的倒数第二个维度可以不一样，但是encoder的qkv维度一样

12.Transformer的并行化提现在哪个地方？
    - Transformer的并行化主要体现在self-attention模块，在Encoder端Transformer可以并行处理整个序列，并得到整个输入序列经过Encoder端的输出，但是rnn只能从前到后的执行

13.Decoder端可以做并行化吗？
    - 训练的时候可以，但是交互的时候不可以

14.简单描述一下wordpiece model 和 byte pair encoding，有实际应用过吗？
    - “传统词表示方法无法很好的处理未知或罕见的词汇（OOV问题）传统词tokenization方法不利于模型学习词缀之间的关系”BPE（字节对编码）或二元编码是一种简单的数据压缩形式，其中最常见的一对连续字节数据被替换为该数据中不存在的字节。后期使用时需要一个替换表来重建原始数据。优点：可以有效地平衡词汇表大小和步数（编码句子所需的token次数）。缺点：基于贪婪和确定的符号替换，不能提供带概率的多个分片结果

15.Transformer训练的时候学习率是如何设定的？Dropout是如何设定的，位置在哪里？Dropout 在测试的需要有什么需要注意的吗？
    - LN是为了解决梯度消失的问题，dropout是为了解决过拟合的问题。在embedding后面加LN有利于embedding matrix的收敛

16.bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？
    - BERT和transformer的目标不一致，bert是语言的预训练模型，需要充分考虑上下文的关系，而transformer主要考虑句子中第i个元素与前i-1个元素的关系
        
8. Transformer的Encoder端和Decoder端是如何进行交互的？和一般的seq2seq有什么差别
    
9. 不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵，会有什么问题

6. 关于 mask 机制，在 Transformer 中有 attention、encoder 和 decoder 中不同的应用区别是什么
    
2. Transformer的复杂度，如何降低复杂度
    - todo
---
# Filtered References
- [1.The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [2.再回首 DeepLearning 遇见了 Transformer](https://mp.weixin.qq.com/s/a3sbbCYioAPkK471BRBPyw)
    - 配图中错误：注意力计算中，根号`dk`不是表示`head`个数，而是矩阵`K`的维度
- [超细节的BERT/Transformer知识点](https://zhuanlan.zhihu.com/p/132554155)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)