- 1、在NLP领域中，LN比BN更合适，为什么？
    - 如果我们将一批文本组成一个batch，那么BN的操作方向是，对每句话的第一个词进行操作。但语言文本的复杂性是很高的，任何一个词都有可能放在初始位置，且词序可能并不影响我们对句子的理解。而BN是针对每个位置进行缩放，这不符合NLP的规律。
    - 由于一个mini batch中的每个句子长度不一致，存在paddding，对列缩放的话会造成误差。
    
- 2、transformer是如何捕获序列中的顺序信息呢？
    - 通过使用 Transformer 我们可以得到一个对于输入信息的 embedding vector，要给模型增加捕获序列顺序的能力，我们创建一个和输入序列等长的新序列，这个序列里包含序列中的顺序信息，我们把这个序列和原有序列进行相加，从而得到输入到 Transformer 的序列。那应该怎样表示序列中的位置信息呢？
    - position embeddings 对每一个位置生成一个向量，然后我们使用模型的学习能力来学习到这些位置的 vector。
    - position encodings 选择一个 function 来生成每个位置的 vector，并且让模型网络去找出该如何去理解这些 encoding vector。这样做的好处是，对于一个选择的比较好的function，网络模型能够处理那些在训练阶段没有见过的序列位置 vector（虽然这也并不是说这些没见过的位置 vector 一定能够表现的很好，但是好在是我们可以有比较直接的方法来测试他们）。
    
- 3、关于 mask 机制，在 Transformer 中有 attention、encoder 和 decoder 中不同的应用区别是什么？

- 4、BERT非线性的来源在哪里？
    - 前馈层的gelu激活函数和self-attention，self-attention是非线性的(计算score的点积操作)
    
- 5、Transformer的点积模型做缩放的原因是什么？
    - 极大的点积值将整个 softmax 推向梯度平缓区，使得收敛困难。
    - 将方差控制为1，也就有效地控制了前面提到的梯度消失的问题。
    - 引申：为什么在其他 softmax 的应用场景，不需要做 scaled
    
- 6、self-attention相比lstm优点是什么？
    - bert通过使用self-attention + position embedding对序列进行编码，lstm的计算过程是从左到右从上到下（如果是多层lstm的话），后一个时间节点的emb需要等前面的算完，而bert这种方式相当于并行计算。
    
- 7、bert的mask为何不学习transformer在attention处进行屏蔽score的技巧？
    - https://www.zhihu.com/question/318355038
    
- 8、BN和LN的区别

- 9、BN训练和测试时的参数是一样的嘛？
    - 对于BN，在训练时，是对每一批的训练数据进行归一化，也即用每一批数据的均值和方差。而在测试时，比如进行一个样本的预测，就并没有batch的概念，因此，这个时候用的均值和方差是全量训练数据的均值和方差，这个可以通过移动平均法求得。
    - 对于BN，当一个模型训练完成之后，它的所有参数都确定了，包括均值和方差，gamma和bata。
- 10、在BERT中，token分3种情况做mask，分别的作用是什么？

- 11、BERT的输入是什么，哪些是必须的，为什么position id不用给，type_id 和 attention_mask没有给定的时候，默认会是什么
- 12、为什么说ELMO是伪双向，BERT是真双向？产生这种差异的原因是什么？
- 13、BERT和Transformer Encoder的差异有哪些？做出这些差异化的目的是什么？
- 14、BERT 的两个任务 Masked LM 任务和 Next Sentence Prediction 任务是先后训练的还是交替训练的
- 15、Transformer的Encoder端和Decoder端是如何进行交互的？和一般的seq2seq有什么差别？
- 16、transformer结构，transformer 与lstm区别
- 17、self-attention attention
- 18、认为测试指标和线上指标差异的原因(分析能力)
- 19、为什么relu比sigmoid更能解决梯度消失
- 20、MSE和交叉熵区别
    - MSE无差别的关注全部类别上预测概率和真实概率的差.
    - 交叉熵关注的是正确类别的预测概率.
- 21、bert的位置编码是什么？
- 22、神经网络的梯度下降介绍下
- 23、过拟合
	- Regularization：数据量比较小会导致模型过拟合, 使得训练误差很小而测试误差特别大。通过在Loss Function 后面加上正则项可以抑制过拟合的产生。缺点是引入了一个需要手动调整的hyper-parameter。
	- Dropout：这也是一种正则化手段，不过跟以上不同的是它通过随机将部分神经元的输出置零来实现。
	- Unsupervised Pre-training：用Auto-Encoder或者RBM的卷积形式一层一层地做无监督预训练, 最后加上分类层做有监督的Fine-Tuning。
	- Transfer Learning（迁移学习）：在某些情况下，训练集的收集可能非常困难或代价高昂。因此，有必要创造出某种高性能学习机（learner），使得它们能够基于从其他领域易于获得的数据上进行训练，并能够在对另一领域的数据进行预测时表现优异。


- 24、crf和hmm区别
- 25、LM和MLM区别