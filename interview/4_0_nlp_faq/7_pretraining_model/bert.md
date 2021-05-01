1. SimCSL是如何获取正例对的？
    - **因为BERT内部每次dropout都随机会生成一个不同的dropout mask**。所以SimCSL不需要改变原始BERT，只需要将同一个句子喂给模型两次，得到的两个向量就是应用两次不同dropout mask的结果。然后将两个向量作为正例对。
    
1. BERT非线性的来源在哪里？
    - 前馈层的gelu激活函数和self-attention，self-attention是非线性的(计算score的点积操作)
    
2. bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？
    - BERT 和 Transformer 的目标不是一致的；BERT 是一个语言模型的预训练模型，它考虑到要充分利用文本的上下文信息；Transformer 的任务是 seq2seq，序列第 i 个位置要充分利用到前 i - 1 个元素的信息，而与该位置之后的其他位置的信息无关。
    
3. 在BERT中，token分3种情况做mask，分别的作用是什么？
    - todo
    
4. BERT的输入是什么，哪些是必须的，为什么position id不用给，type_id和 attention_mask没有给定的时候，默认会是什么？
    - todo
    
5. 为什么说ELMO是伪双向，BERT是真双向？产生这种差异的原因是什么？
    - todo
    
6. BERT和Transformer Encoder的差异有哪些？做出这些差异化的目的是什么？
    - todo
    
7. BERT 的两个任务 Masked LM 任务和 Next Sentence Prediction 任务是先后训练的还是交替训练的？
    - todo
    
8. bert的位置编码是什么
    - todo
    
### todo
- 用法
- 蒸馏