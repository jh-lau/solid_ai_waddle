# SpERT: Span-based Entity and Relation Transformer
PyTorch code for SpERT: "Span-based Entity and Relation Transformer". 
[https://arxiv.org/abs/1909.07755](https://arxiv.org/abs/1909.07755)
(accepted at ECAI 2020).

![alt text](data_path/spert.png)

## 环境配置
### Requirements
- Python 3.5+
- PyTorch 1.1.0+ (tested with version 1.3.1)
- transformers 2.2.0+ (tested with version 2.2.0)
- scikit-learn (tested with version 0.21.3)
- tqdm (tested with version 4.19.5)
- numpy (tested with version 1.17.4)
- 可选
  - jinja2 (tested with version 2.10.3) - 用以可视化模型抽取结果
  - tensorboardX (tested with version 1.6) - 用以保存训练结果到tensorboard
  
#### 模型调用
- 修改parameters.yaml文件中前两行文件路径的位置：分别为已训练模型文件路径和spo数据类型json文件路径。
- 调用service.inference()方法即可。

#### 数据说明
- `*.xls`文件为原始`excel`语料或者是手动标注的`execel`语料
- `*_temp.json`为`excel`处理成的一对多的`json`文件
- `*_final.json`为上一步中对文本进行分句分词（单字），对spo列表进行索引处理后符合模型输入的最终数据
- 数据增强方式
  - 实体替换
    - 基于字典
  - 非实体替换
    - 基于词向量的同义词替换：分词后，逐一通过word2vec等工具进行非实体的词语片段替换，缺点是基本句式无法改变，依赖word2vec语料的质量，否则会导致句子语义不完整。当然对于对句子语义不要求的任务（如句子分类）等影响较小。
    - 基于反向翻译的同义词替换：需要人工确认实体是否被错误翻译，通常体现在人名上，优点在于可以改变句式，且基本能保持语义通顺，依赖翻译api接口。
    - 随机操作：对非实体部分的随机操作，同样会影响语义，且对句式结构的影响不大。
      - 随机增加
      - 随机删除
      - 随机替换
