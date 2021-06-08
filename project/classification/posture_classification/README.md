# 动作分类

### 数据准备：训练数据及格式要求
 - 1.拍摄训练视频
 - 2.调用`utils.video2frames`对视频进行图像转换，具体转换帧率见方法说明
 - 3.对转换的图像打上标签
    - 将分帧的图像分类到包含`positive`和`negative`的文件夹中
    - 在`positive`文件夹中
    - 在`negative`文件夹中，直接命名图片名称即可，如`pics01.jpg`
 - 4.调用`utils.utils.data_to_csv`对数据集进行json格式整理，作为模型读入训练数据的文件对象，具体地，json中列表中每个训练数据字典的格式为：`filename,label,score`
 
运行`fetch_data.sh`可以执行模型下载操作（本项目训练数据未上传到ftp服务器）。下载的文件夹结构类似：
```
data_path
  - model
    - model.h5
```

### 训练与推理
#### 训练

- 调用`service.py`文件进行训练，设置见文件

#### 测试推理

- 调用`service.py`文件进行单张图片测试，设置见文件

### 模型推理识别效果

- 运行`demo.py`文件进行视频播放以及模型实时推测展示。更改`config.yaml`文件中的模型路径进行选择不同模型，更改`demo.py`文件中主函数的视频路径选择不同的视频进行预测。


### BUG
- 使用预训练权重时，当`fit_generator(data_gen)`与`evaluate_generator(data_gen)`使用同样的数据时，训练期间的如果准确率为90%，在评估的时候准确率几乎等于没训练过的随机效果的情况，踩坑可能的原因（目前没有完全确定根源），参考[Keras官方issue](https://github.com/keras-team/keras/issues/6499)与[stackoverflow处理方法](https://stackoverflow.com/questions/55569181/why-is-accuracy-from-fit-generator-different-to-that-from-evaluate-generator-in?noredirect=1&lq=1)：
    - 在`flow_from_dataframe`中设置seed
    - 在`flow_from_dataframe`中`shuffle`设置为`False`
    - 使用API`ImageDataGenerator`时，不使用`rescale`而使用`preprocess_input`作为预处理函数
    - `fit_generator`中`use_multiprocessing`为`False`，`workers=1`
    - 网络结构中有`batch-normalization`层需要将其设置为可训练：`trainable=True`


### Notices
- 数据预处理读取图片的时候，如果读出的图片是按照`rgb`通道读取的，那么对新图片做预测时，新图片也需要先转换成`rgb`格式，通常在使用`opencv`工具时需要注意该问题，因为其默认读取的图片格式是`bgr`通道。
