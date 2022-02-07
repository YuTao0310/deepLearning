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