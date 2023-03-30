# 模型加载

## 1 CPU模型

linux下使用cpu训练好的模型能够在windows被加载；

```python
from ctgan.synthesizers.base import BaseSynthesizer

ctgan = BaseSynthesizer.load('./ctgan/ctgan_aliyun.pt')

samples = ctgan.sample(1000)
```

windows下使用cpu训练好的模型能够在linux被加载。

```python
from ctgan.synthesizers.base import BaseSynthesizer

ctgan = BaseSynthesizer.load('ctgan_windows.pt')

samples = ctgan.sample(1000)
```

## 2 GPU模型

linux下使用gpu训练好的模型能够在windows被加载；

```python
from ctgan.synthesizers.base import BaseSynthesizer

ctgan = BaseSynthesizer.load('./ctgan/ctgan_linux.pt')

samples = ctgan.sample(1000)
```

linux使用gpu训练好的模型不能在windows加载的原因在于ctgan的版本不符合。例如，linux是ctgan=0.4.0下训练的，而windows的ctgan版本为0.5.0，这样windows当然无法加载正确的模型。

```shell
'NoneType' object is not iterable
```

# 查看依赖关系

## 未安装包

法1：前往pypi下载包的源代码，其中setup.py中描述了包的详细的依赖关系。这种方法能够获得各种各样的版本的依赖关系。

法2：前往包的github网址，其中setup.py文件描述了详细依赖关系。这种方法只能查到最新版本的依赖关系。

## 安装包

法1：前往site-packages/xxxx-x.x.x.dist-info文件夹下，其中METADATA文件描述了包的依赖关系。

法2：安装pipdeptree，在终端中输入pipdeptree，可以显示当前环境下所有包的依赖关系树状图。

# torch torchvision cuda版本

## cuda驱动

Driver API(nvidia-smi显示的)以及Runtime API(nvcc --version显示的)
安装torch版本中对应的cuda版本为Runtime API。

## 不同版本对应

torch 1.7.0对应torchvision 0.8.0 0.8.1

torch 1.7.1对应torchvision 0.8.2

ctgan 0.4.3能与torch1.7.0 1.7.1兼容

sdv0.12.1能与以上兼容

**我选择的python版本为3.8.1**

==torch1.7.1+cpu==

```
certifi==2021.10.8
numpy==1.22.3
Pillow==9.0.1
torch==1.7.1+cpu 
torchaudio==0.7.2
torchvision==0.8.2+cpu 
typing_extensions==4.1.1
wincertstore==0.2
```

==cuda11.0直接对应的版本==

```
certifi==2021.10.8
numpy==1.22.3
Pillow==9.0.1
torch==1.7.1+cu110
torchaudio==0.7.2
torchvision==0.8.2+cu110
typing_extensions==4.1.1
```

==cuda11.0能安装cuda11.3对应的版本==

```
certifi==2021.10.8
charset-normalizer==2.0.12
idna==3.3
numpy==1.22.3
Pillow==9.0.1
requests==2.27.1
torch==1.11.0+cu113
torchaudio==0.11.0+cu113
torchvision==0.12.0+cu113
typing_extensions==4.1.1
urllib3==1.26.9
```

==nvcc为10.1 10.2安装==

`conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch`

**torch测试代码**

```python
import torch
flag = torch.cuda.is_available()
print(flag)

ngpu= 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3,3).cuda()) 
```

# 日志

## 1、logging模块

日志级别（由高到低）：

![image-20220113144739422](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220113144739422.png)

具体代码：

```python
import logging
logging.basicConfig(filename = './sdv/logging_example.log', level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logging.debug('debug message')
logging.info('info message')
logging.warning('warn message')
logging.error('error message')
logging.critical('critical message')
```

## 2、将输出信息存到一个文件中

```python
import sys
sys.stdout = open('./ctgan/ctgan_epoch300_record.log', mode = 'w',encoding='utf-8')
```

# SDV评估生成数据的指标

[Single Table Metrics](https://sdv.dev/SDV/user_guides/evaluation/single_table_metrics.html)

## 1、**Statistical Metrics**

`sdv.metrics.tabular.KSTest`
`sdv.metrics.tabular.CSTest`

## 2、**Likelihood Metrics**

`sdv.metrics.tabular.BNLikelihood`
`sdv.metrics.tabular.BNLogLikelihood`
`sdv.metrics.tabular.GMLogLikelihood`

## 3、**Detection Metrics**

`sdv.metrics.tabular.LogisticDetection`
`sdv.metrics.tabular.SVCDetection`

## 4、**Machine Learning Efficacy Metrics**

- Binary Classification Metrics:
  - `BinaryDecisionTreeClassifier`
  - `BinaryAdaBoostClassifier`
  - `BinaryLogisticRegression`
  - `BinaryMLPClassifier`
- Multiclass Classification Metrics:
  - `MulticlassDecisionTreeClassifier`
  - `MulticlassMLPClassifier`
- Regression Metrics:
  - `LinearRegression`
  - `MLPRegressor`

## 5、**Privacy Metrics**

- Categorical metrics:
  - `sdv.metrics.tabular.CategoricalCAP`
  - `sdv.metrics.tabular.CategoricalZeroCAP`
  - `sdv.metrics.tabular.CategoricalGeneralizedCAP`
  - `sdv.metrics.tabular.CategoricalKNN`
  - `sdv.metrics.tabular.CategoricalNB`
  - `sdv.metrics.tabular.CategoricalRF`
  - `sdv.metrics.tabular.CategoricalEnsemble`
- Numerical metrics:
  - `sdv.metrics.tabular.NumericalMLP`
  - `sdv.metrics.tabular.NumericalLR`
  - `sdv.metrics.tabular.NumericalSVR`
  - `sdv.metrics.tabular.NumericalRadiusNearestNeighbor`

# torch用法

## 读入图片

* PIL的Image.open，得到H*W*C，为PIL Image对象
* cv2.imread， 为ndarray对象，像素格式，相当于PIL转化为ndarray
* torch.utils的read_image，得到CHW，为Tensor对象
* plt.show(image)中image必须为numpy类型，且为HWC形式，为ndarray对象。

图片能够以PIL image numpy.ndarray torch.tensor三种形式存在，三者之间能够相互转换，也能够进行相应的操作。

## torchvision.transforms.ToPILImage和torchvision.transforms.functional.to_pil_image

这两个功能相同，还有类似的模块功能也差不多。

## 模型库

* torchvision.datasets
* torch.hub.load(path)
* timm.models(pip install timm即可)
* 直接下好加载，分为加载字典和加载整个模型（前者比后者小）

torh.load本地加载 模型或者模型字典
torch.hub.load（可以从github加载）、torch.hub.load_state_dict_frome_url（给定网址，后面还需结合model.load_state_dict或者model._load_from_state_dict才能加载完整模型）远程加载

torchvision.datasets以及timm.models都是用pytorch框架实现，构建模型时都利用了torch.hub的相关函数。

## 修改尺寸

* torch.transpose
* torch.reshape
* torch.view
* torch.permute

torch.contiguous经常与上述一起使用

is_contiguous直观的解释是**Tensor底层一维数组元素的存储顺序与Tensor按行优先一维展开的元素顺序是否一致**。

Tensor多维数组底层实现是使用一块连续内存的1维数组（行优先顺序存储，下文描述），Tensor在元信息里保存了多维数组的形状，在访问元素时，通过多维度索引转化成1维数组相对于数组起始位置的偏移量即可找到对应的数据。某些Tensor操作（如transpose、permute、narrow、expand）与原Tensor是共享内存中的数据，不会改变底层数组的存储，但原来在语义上相邻、内存里也相邻的元素在执行这样的操作后，**在语义上相邻，但在内存不相邻**，即不连续了（is not contiguous）。

## 在线数据增强 datasets dataloader

torch默认实现是在线数据增强的，每次从dataloader中加载数据时，dataloader会调用fetcher类中的fetch函数

```python
class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super(_IterableDatasetFetcher, self).__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (self.drop_last and len(data) < len(possibly_batched_index)):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self.collate_fn(data)
```

`data = next(self.dataset_iter)`相当于调用下列函数，__getitem__是先获取原图片再调用transforms进行转换。

```python
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
```

这也就意味着，每次遍历dataloader，会对原图进行transforms操作。如果transforms中含有数据增强随机算子，每次获得的图片有可能不一样，这样的话，只要epochs足够大，能够实现在线数据增强，也就是虽然图片并没有直接扩充，但效果上相当于扩充了图像数量。

# 深度学习总框架

## 人工智能 机器学习 深度学习

人工智能方法包括机器学习，机器学习方法包括传统机器学习和深度学习。

## 深度学习囊括内容

### 总分类

监督学习 supervised learning
半监督学习 self-supervised learning
无监督学习 unsupervised learning

### 方法论（methodology）

* 数据增强 data augmentation

* 迁移学习 transfer learning
  
  - transfer learning最小化当前的model（只有一个）在所有任务上的loss，所以model pretraining希望找到一个在所有任务（实际情况往往是大多数任务）上都表现较好的一个初始化参数，这个参数要在多数任务上**当前表现较好**。
  - meta learning最小化每一个子任务训练一步之后，第二次计算出的loss，用第二步的gradient更新meta网络，这代表了什么呢？子任务从【状态0】，到【状态1】，我们希望状态1的loss小，说明meta learning更care的是**初始化参数未来的潜力**。
  
  前者关注当下，后者关注未来。
  
  参考：https://zhuanlan.zhihu.com/p/136975128
  
  迁移学习方式：
  
  * CNN feature + classifier 利用深度学习模型的已经训练好的特征提取器，参数固定直接使用，应用在下游任务过程中，只需要优化classifier中的参数即可。
  
  * Fine-tuning（微调）：
    
    1）微调所有层
    
    2）固定网络前面几层权重，只微调网络的后面几层。这样做两个原因：其一，避免数据量过小造成过拟合的现象；其二，CNN前面几层的特征中包含更多的一般的特征（比如边缘信息、色彩信息），但是CNN后面几层的特征学习更加注重高层信息，也就是语义特征，这与特定数据集有关系。

* 元学习 meta learning

* 域自适应 domain adaptation
  
  * 域自适应倾向于解决特征空间和类别空间一致，但是特征分布不一致的任务。举个简单的例子：对于同样一个目标检测的任务，基于**公开数据集(源域)**训练出了一个模型，由于公开数据集与**自己采集的数据集(目标域)**的特征分布存在差异，那么模型可能会在源域上过拟合，导致在目标域上测试效果不好。但是此时源域与目标域对应的都是同一个目标检测任务，且二者对应的特征空间和类别空间是一样的，那么就可以通过域自适应方法将源域模型迁移到目标域上。常用的域自适应方法如样本自适应、特征自适应以及模型自适应可以从任务的不同阶段提取源域与目标域的相似性关系，提升在目标域上的泛化性。
  
  * 相比于域自适应，迁移学习的研究范围更广，可以用于特征空间和类别空间不一致的情况，即对于两种不同的任务，迁移学习也可以利用实现相似的领域知识进行迁移。从这个层面上来讲，**域自适应可以看做是迁移学习的一个子方向**。
    
    参考：https://www.zhihu.com/question/374562547/answer/1243077910

* 小样本学习 few-shot learning
  
  小样本学习是元学习在监督学习上的应用。

* 特征工程 feature engineering

* 表征学习 representation learning

Representation learning is concerned with training machine learning algorithms to learn useful representations, e.g. those that are interpretable, have latent features, or can be used for transfer learning.

Deep neural networks can be considered representation learning models that typically encode information which is projected into a different subspace. These representations are then usually passed on to a linear classifier to, for instance, train a classifier.

Bengio的定义是：

learning representations of the data that make it easier to extract useful information when building classifiers or other predictors

### 根据应用场景分类

* 计算机视觉 computer vision
  
  分类、定位、检测、分割（语义和实例）、图像生成、视频、超分等等

* 自然语言处理 natural language processing

* 时序数据 time series
  
  speech、audio

* 表格数据 tabular data

### 根据应用领域分类

工业、医疗、农业、交通等等

# 分类模型

## 1、参考链接

https://pytorch.org/vision/stable/models.html#classification
https://paperswithcode.com/methods/category/image-models

## 2、发展历史

[1、AlexNet CNN 2012](https://zhuanlan.zhihu.com/p/42914388)
[2、VGG CNN 2014](https://zhuanlan.zhihu.com/p/41423739)
[3、GoogleNet(Inception V1) CNN 2014](https://zhuanlan.zhihu.com/p/52802896)
[4、Inception V3 CNN CNN 2015](https://zhuanlan.zhihu.com/p/52802896)
[5、ResNet CNN 2015](https://zhuanlan.zhihu.com/p/31852747)
[6、SqueezeNet CNN 2016](https://zhuanlan.zhihu.com/p/49465950)
[7、Wide Resnet CNN 2016]()
[8、ResNext CNN 2016](https://zhuanlan.zhihu.com/p/51075096)
[9、DenseNet CNN 2017](https://zhuanlan.zhihu.com/p/37189203)
[10、ShuffleNet CNN 2017](https://zhuanlan.zhihu.com/p/32304419)
[10、ShuffleNetV2 CNN 2018]()
[11、MobileNet V1 V2 V3 CNN 2017 2018 2019](https://zhuanlan.zhihu.com/p/70703846)
[12、EfficientNet V1 V2 CNN 2019 2021](https://zhuanlan.zhihu.com/p/67834114)
[13、RegNet CNN 2020](https://www.zhihu.com/question/384255803             )
[14、VisionTransformer Attention 2020](https://zhuanlan.zhihu.com/p/340149804 https://zhuanlan.zhihu.com/p/317756159)
[15、DeiT Attention 2021](https://zhuanlan.zhihu.com/p/394627382)
[16、SwinTransformer Attention 2021](https://zhuanlan.zhihu.com/p/367111046)
[17、ConvNeXt CNN 2022](https://zhuanlan.zhihu.com/p/458016349)

# 目标检测模型

## 1、参考链接

https://paperswithcode.com/methods/category/object-detection-models
https://paperswithcode.com/methods/category/one-stage-object-detection-models
https://pytorch.org/vision/stable/models.html#object-detection-instance-segmentation-and-person-keypoint-detection

# 算法改进思路

## 1、网络结构改进

* 重新设计网络架构的模式：
  
  网络结构模式分为同质架构（isotropic architecture）或者非结构性架构（non-hierarchical architecture）、金字塔架构（pyramidal architecture）或者结构化架构（hierarchical architecture）。[参考](https://zhuanlan.zhihu.com/p/455086818)
  
  同质架构由相同的blocks串联而成，基于同质架构的模型只需要用特征大小（patch embedding的维度）和网络深度（blocks的数量）两个参数定义，比如说ViT、MLP-Mixer。
  
  金字塔架构包含几个不同的stage，各个stage之间是一个下采样操作（比如stride为2的max pooling或者conv、Swin Transformer的Parch Merging）。金字塔架构的模型例子有Resnet、Swin Transformer、ConvNeXt。

* 引入新模块（例如SENet）

* 多尺度信息

* 增加网络深度、广度、图片分辨率（Efficient说明了三点对精度的影响）

* 改变归一化方式：BN替代dropout（但swin transformer中的MLP有dropout），LN替代BN。归一化的过程其实就是正则化的过程。

* 改变激活函数

* 改变卷积核大小和个数

![preview](https://pic1.zhimg.com/v2-43d750b4a1895747242ebb46247811d4_r.jpg)

## 2、训练策略的优化（trick）

* 是否引入数据增强策略（正则化）：通常在dataset中的transform中进行设定；mix_up数据增强策略需要在训练阶段进行小幅度的修改

* 搜索算法改进：优化算法改进；采用NAS（神经网络结构搜索，自行构造网络结构）搜索参数

* 损失函数改进：CrossEntropyLoss、LabelSmoothingCrossEntropy（正则化）、SoftTargetCrossEntropy（当有mixup数据增强策略时，需要使用该损失函数）

* optimzier优化器：选择不同的优化器；

* 优化参数过程中是否引入weight_decay（正则化）

* training schedule：学习率恒定不变、warmup策略、随着训练过程变化（比如在训练到1/3 2/3时学习率发生变化；直接使用cosine的学习率变化）；epoch或者steps的设定

* 其他超参数设定：batch_size、input_size（resolution）、优化器中的参数（如momentum、beltas）

# 机器学习与深度学习通用问题

## 模型过拟合

主要表现为：训练集效果好，测试集效果差（泛化能力弱）。

主要原因是：数据太少或者模型太复杂

解决方法有：

* 数据上，从数据源头增加数据；采用传统数据增强或者GAN来进行增强
* 使用合适的模型：包括设计合适的空间结构以及选择合适的训练策略
* 结合多种模型
* 使用正则化手段：其实数据增强、dropout、weight decay、BN等等都是正则化手段

## convariate shift

当训练集与测试集的分布不相同时，模型精度会出现大幅下降。

经典的机器学习模型中，我们习惯性假设训练数据集和目标训练集有着相同的概率分布。而在现实生活中，这种约束性假设很难实现。当训练数据集和测试集有着巨大差异时，很容易出现过拟合的现象，使得训练的模型在测试集上表现不理想。

解决方法：

1、采用领域自适应方法（迁移学习的一种特殊情况）

2、扩充数据集，增加数据集的多样性

## 参数影响

* bach-size

batch-size越大，增加内存利用率，降低内存容量；对相同数据量的处理速度加快，一次epoch，迭代次数减小，达到相同精度所需时间增大；梯度下降方向越准，引起训练振荡越小，但过大，其下降方向基本不变，可能会陷入局部最优。

batch-size越小，降低内存利用率，增加内存容量；对相同数据量的处理速度变慢，达到相同精度所需时间可能会减小；不容易收敛，但会引入随机性。

## 参数量 计算量 模型大小 推理速度

模型复杂度由参数量和计算量两个层次描述

参数量指参数的个数，可以认为是存储参数的空间复杂度

计算量（模型计算力）指耗费计算资源的多少，通常一FOPS为单位，1TOPS=$10^3$GOPS=$10^6$MOPS=$10^9$KOPS=$10^{12}$FLOPS

模型大小包括参数所占空间、中间数据体尺寸、各种零散的内存占用（如成批的数量数据、扩充的数据）；也有说法用FLOPS表示（convnext采用此法）

推理速度（inference time）可以用fps或者throughout表示，GPU的推理速度不仅受计算量影响，而且受仿存带宽的影响。在仿存带宽不变的情况下，受仿存量的影响，仿存量与batchsize、空间尺寸（H W）以及网络宽度（通道数）。

衡量GPU的主要考虑三个点：

1、算力FLOPS（具体看频率以及CUDA核心数【针对Nvidia显卡】）
2、仿存带宽。越大，读取速度越快。
3、显存大小。显存越大，能够承受的batch数目更多，单块GPU同时处理图片的数目越多。

## 网络宽度和网络深度

对于卷积层来说，宽度是指输出维度（通道）。对于一个网络来说，宽度是指所有参数层的总体输出维度，而深度是指参数层（卷积层、全连接层）的层数。

## 特殊卷积

* Group Conv

当分组数与通道数相同时，变成depth-wise conv

* 深度可分离卷积 depthwise separable convolution

包含depth-wise conv（对通道进行卷积） 以及 point-wise conv（1*1卷积核）

* channel shuffle

参数量比深度可分离卷积更少

* Dilated/Atrous Convolution 空洞卷积

扩大感受野以及捕获多尺度上下文信息

# 可视化

## 特征图可视化

参考2022-04-01的组会PPT内容

## 网络结构可视化

### 分类

* 网络结构层用模块表示，结果层（每一层网络的输出结果）在传输线上

1、swin transformer论文原图

![image-20220413132856607](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413132856607.png)

2、convnext原图

![image-20220413133020880](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413133020880.png)

3、inception

![preview](https://pic4.zhimg.com/v2-660fdabd306652c32afe7ce15bd9d38b_r.jpg)

4、Identification method of vegetable diseases based on transfer learning and
attention mechanism

![image-20220413134806913](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413134806913.png)

* 结果层用模块表示，网络结构层在传输线上

1、alexnet

![img](https://pic2.zhimg.com/80/v2-3f5a7ab9bcb15004d5a08fdf71e6a775_720w.jpg)

2、Identification method of vegetable diseases based on transfer learning and
attention mechanism

![image-20220413134716872](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413134716872.png)

3、Local Relation Networks for Image Recognition

![image-20220413134945162](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413134945162.png)

* 网络结构层和结构层均用模块表示

1、Assessment of state-of-the-art deep learning based citrus disease detection techniques using annotated optical leaf images

![image-20220413133232363](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413133232363.png)

2、Autonomous Mobile Robot for Apple Plant Disease Detection based on
CNN and Multi-Spectral Vision System

![image-20220413133251494](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413133251494.png)

3、Leaf spot attention network for apple leaf disease identification

![image-20220413133114675](C:\Users\28439\AppData\Roaming\Typora\typora-user-images\image-20220413133114675.png)

每个模块既包含网络结构层又某种意义上表示着结果层。

### 工具

* https://github.com/dair-ai/ml-visuals
* 

# 开题报告

## PPT

1、注意整理逻辑，国内外现状4个板块之间的联系要么在PPT中展示，要么在演讲时展示。

2、关键内容记得加粗或者变颜色展示。

3、论文成果可以附在研究方案下方，这样更加醒目。

4、一页内容不能放太多，如果实在太多，记得加粗，或者采用流程框图的形式展示，切忌太多文字。

5、研究方案记得使用动词描述，这样更加生动。

6、内容之间记得一一对应，前面提及到的，后面记得用到。********
