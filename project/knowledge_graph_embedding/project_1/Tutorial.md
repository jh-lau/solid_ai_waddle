# 开发者文档
# [技术文档](Technical_Prerequisite.md)
### 已实现模型
 - [x] RotatE
 - [x] TransE
 - [x] ComplEx
 - [x] DistMult
 
### 评估指标
 - [x] MRR, MR, HITS@1, HITS@3, HITS@10 

### 训练数据及格式要求
 - *entities.dict*: 实体字典，每行格式为：id entity
 - *relations.dict*: 关系字典，每行格式为: id relation
 - *train.txt*: 训练三元组，每行格式为：头实体 关系 尾实体（注：分隔符不一定为tab）
 - *valid.txt*: 验证三元组数据，格式同上
 - *test.txt*: 测试三元组数据，格式同上
 
如果获取的语料只有三元组数据，可通过调用项目下的`ops.utils.get_entities_relations_dict`方法自动生成以上训练所需字典和分割后的数据，具体用法见方法说明。

运行`fetch_data.sh`可以执行数据下载操作。下载的文件夹结构类似：
```
data_path
  - data
    - cn_baike_300k
      - entities.dict
      - relations.dict
      - train.txt
      - valid.txt
      - test.txt

  - model
    - RotatE_cn_baike_300k
      - checkpoint
      - config.json
      - entity_embedding.npy
      - relation_embedding.npy
      - test.log
      - train.log
```

### 训练与测试
#### 训练

**方法1：命令行方式运行**
```
CUDA_VISIBLE_DEVICES=0 python -u core/train.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data_path/data/cn_baike \
 --model RotatE \
 -n 256 -b 1024 -d 1000 \
 -g 24.0 -a 1.0 -adv \
 -lr 0.0001 --max_steps 150000 \
 -save data_path/models/RotatE_cn_baike_0 --test_batch_size 16 -de
```
**方法2：脚本运行（命令行方式的shell封装，推荐使用）**
```
bash train.sh train RotatE cn_baike 0 0 512 64 1000 0.1 1.0 0.000002 40000 8 -de
```
#### 测试

通过命令行测试

```CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u core/train.py --do_test --cuda -init $SAVE```

或者通过脚本运行，脚本对应的位置参数详见脚本。

```bash train.sh valid/test cn_baike_300k 0 13```

### 模型训练参数说明（参数默认值参见core/train.py文件）
| 参数 | 说明 |
| ---- | ---- | 
| --cuda | 是否使用gpu | 
| --do_train | 训练模式 | 
| --do_valid | 验证模式 | 
| --do_test | 测试模式 | 
| --evaluation_train | 对训练数据进行验证 | 
| --data_path | 训练数据路径 | 
| --model | 模型名称（TransE、RotatE等） | 
| --double_entity_embedding| 实体向量维度加倍（某些模型不适用）|
| --double_relation_embedding| 关系向量维度加倍（某些模型不适用）|
| --negative_sample_size| 负采样规模 |
| --hidden_dim| 实体和关系的训练结果向量维度 |
| --gamma| 公式参数 |
| --negative_adversarial_sampling| 是否适应自对抗负采样技术 |
| --adversarial_temperature| 自对抗负采样系数 |
| --batch_size| mini-batch参数 |
| --regularization| 正则项系数 |
| --test_batch_size | 测试batch-size大小 |
| --uni-weight| 采样权重 |
| --learning_rate| 学习速率 |
| --cpu_num| 参与数据处理的cpu数量 |
| --init_checkpoint| 模型缓存文件路径 |
| --save_path| 模型保存路径 |
| --max_steps| 最大训练迭代次数 |
| --warm_up_steps| 学习速率调节参数 |
| --save_checkpoint_steps| 训练保存模型间隔 |
| --valid_steps| 模型验证间隔 |
| --log_steps| 日志输出间隔 |
| --test_log_steps| 测试日志输出间隔 |
| --nentity| 实体数量，非手动设置 |
| --nrelation| 关系数量，非手动设置 |

### 通用数据集复现结果（RotatE）
| 数据集| FB15k | FB15k-237 | wn18 | wn18rr | YAGO3-10 |
|--------|---------|--------|--------|--------| ---------|
| 实体规模| 14951 | 14541 | 40943 | 40943 | 123182 |
| 关系规模 | 1345 | 237 | 18 | 11 | 37 |
| MRR | .786 | .336 | .949|.476 | .49 |
| MR | 42 | 178 | 259 | 3342 | 1886 |
| HITS@1 | .732 | .239 | .943 | .428 | .549 |
| HITS@3 | .822 | .375 | .952 | .495 | .674 |
| HITS@10 | .879 | .53 | .959 | .572 | .674 |

### 自建数据集训练结果（未特殊说明为RotatE模型结果）
| 数据集|cn_baike | cn_bigcilin | cn_military |cn_military(TransE)|
|--------|---------|--------|--------|-----|
| 实体规模| 159387 | 208817 | 28786 | 28786
| 关系规模 | 8695 | 12063 | 116 | 116 |
| MRR | .269 | .267 | .34 | .33 |
| MR | 36215 | 70827 | 3916 | 3448 |
| HITS@1 | .243 | .253 | .295 | .288 |
| HITS@3 | .282 | .274 | .363 | .359 |
| HITS@10 | .316 | .293 | .43 | .436 |


## 推理接口
模型加载函数`load_model()`参数为已训练模型的文件夹路径，其内必须包含`config.json`配置文件，以保证模型可以被正常加载。

> 本项目的`config.yml`文件因与配置文件的功能重叠，因此暂不使用。

模型推理函数`inference()`参数为`头实体 关系 尾实体`构成的字符串（注意顺序必须严格遵守，中间用空格隔开），结果为预测的头尾实体数量各10个。输入：`'摩耶号/Maya巡洋舰 建造时间 1928年'`,预测结果如下图：
![result](data_path/images/inference_result1.png)

> 推理时间与实体和关系的规模正相关。

*Beta推理接口：service_v2.py*

> 直接使用训练好的实体关系向量进行推理运算，目前只支持`TransE`模型。速度方面：数据加载在`200ms`以内，CPU下运算耗时在`50ms`左右。

模型加载函数`load_model()`参数为已训练模型的文件夹路径与训练数据文件夹路径，其中模型文件夹路径必须包含`config.json`配置文件，以保证模型可以被正常加载。

模型推理函数`inference()`参数为`头实体 关系 尾实体`构成的字符串（注意顺序必须严格遵守，中间用空格隔开）与选择的训练模型，**注：此处模型默认为`TransE`，加载的向量文件必须与训练该向量的模型一致**，结果为预测的头尾实体数量各10个。