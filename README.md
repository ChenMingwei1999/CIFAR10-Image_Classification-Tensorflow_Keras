 # Image Classification：Classical CNN Models Trained on CIFAR-10
 
在本仓库中，通过 Tensorflow 的 Keras 框架搭建了五种经典的卷积神经网络模型，并在 CIFAR-10 数据集上进行训练、测试和评估；使用训练得到的模型，通过 PyQt 开发了一款交互式的图像分类软件；通过调整网络模型的超参数，观察并研究其产生的影响。
 
 # 目录


<!-- TOC -->

- [1 项目简介](#1-项目简介)
  - [1.1 主要工作](#11-主要工作)
  - [1.2 文件结构](#12-文件结构)
  - [1.3 适用场景](#13-适用场景)
- [2 环境配置](#2-环境配置)
  - [2.1 软件环境](#21-软件环境)
  - [2.2 硬件环境](#22-硬件环境)
- [3 数据集](#3-数据集)
- [4 网络模型搭建、训练、测试与评估](#4-网络模型搭建训练测试与评估)
  - [4.1 LeNet-5](#41-lenet-5)
  - [4.2 AlexNet](#42-alexnet)
  - [4.3 VGGNet](#43-vggnet)
  - [4.4 GoogLeNet](#44-googlenet)
  - [4.5 ResNet](#45-resnet)
- [5 CIFAR-10 图像分类软件](#5-cifar-10-图像分类软件)
  - [5.1 使用说明](#51-使用说明)
  - [5.2 测试数据](#52-测试数据)
  - [5.3 常见问题](#53-常见问题)
- [6 超参数研究](#6-超参数研究)
  - [6.1 学习率](#61-学习率)
  - [6.2 优化器](#62-优化器)
  - [6.3 损失函数](#63-损失函数)
  - [6.4 激活函数](#64-激活函数)
  - [6.5 批尺寸](#65-批尺寸)
  - [6.6 回合数](#66-回合数)
- [版权许可](#版权许可)
- [联系作者](#联系作者)
- [更新日志](#更新日志)

<!-- /TOC -->

# 1 项目简介
## 1.1 主要工作
* 通过 Tensorflow 的 Keras 框架搭建了 **LeNet-5**、**AlexNet**、**VGGNet** 、**GoogLeNet**、**ResNet** 五种卷积网络模型
* 在 **CIFAR-10 数据集**上对上述五个网络进行训练、测试和评估，得到 **Accuracy** 和 **Loss** 曲线
* 使用训练得到的模型，通过 PyQt 开发了一款**交互式图像分类软件**
* 调整 GoogLeNet 模型的**学习率**、**优化器**、**损失函数**、**激活函数**、**批尺寸**、**回合数**六种超参数，观察并研究其产生的影响

## 1.2 文件结构
```
├── Readme.md                      // 项目介绍
├── LICENSE                        // 许可证
├── .idea                          // 项目配置信息
├── _pycache_                      // 项目编译信息
├── AlexNet.py                     // AlexNet 网络
├── GoogLeNet.py                   // GoogLeNet 网络
├── icons                          // 图标
├── image                          // 图片
│   ├── evaluation metric_image    // 模型评估图片
│   ├── screenshot_image           // 截屏图片
│   └── test_image                 // 软件测试图片
├── LeNet5.py                      // LeNet5 网络
├── main.py                        // 软件主程序
├── main_ui.py                     // 软件主界面类
├── main_ui.ui                     // 软件主界面设计
├── my_model                       // 模型文件
│   ├── my_AlexNet                 // 训练的 AlexNet 模型
│   ├── my_GoogLeNet               // 训练的 GoogLeNet 模型
│   ├── my_LeNet                   // 训练的 LeNet 模型
│   ├── my_ResNet                  // 训练的 ResNet 模型
│   └── my_VGGNet                  // 训练的 VGGNet 模型
├── ResNet.py                      // ResNet 
├── resource.qrc                   // 软件图标资源文件
├── resource_rc.py                 // 软件图标编译文件
└── VGGNet.py                      // VGGNet 网络
```
## 1.3 适用场景
* 计算机视觉、深度学习领域的初学者学习与探索
* 经典卷积神经网络实现图像分类的项目入门
* CIFAR-10 数据集、Tensorflow 与 Keras 框架的初步了解与学习


# 2 环境配置

## 2.1 软件环境
名称 | 版本 
---------|----------
 conda | 4.10.3 
 CUDA | 10.1.105
 cuDNN | 7.6.5
 keras | 2.6.0
 matplotlib | 3.4.1
 numpy | 1.18.5
 opencv | 4.5.1
 pyqt5 | 5.15.4
 python | 3.8.3 
 tensorflow-gpu | 2.3.0


## 2.2 硬件环境 
  
名称 | 型号
---------|----------
 CPU | 11th Gen Intel(R) Core(TM) i5-1135 G7 @2.40GHz
 GPU | NVIDIA GeForce MX450
 OS | Microsoft Windows 10(x64)



# 3 数据集
CIFAR-10 数据集是一个用于识别普适物体的小型数据集，它包含 **10** 个类别、**60000** 张大小为 **32×32** 的彩色 RGB 图像，每类各 6000 张图。其中测试集共 10000 张，单独构成一批，在每一类中随机取 1000 张单独组成；训练集由剩下的随机排列组成，共 50000 张，构成了 5 个训练批，每一批 10000 张图，值得注意的是，**一个训练批中的各类图像的数量不一定相同**。

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/cifar10.jpg" width=500></div>

* [CIFAR-10 数据集官网](http://www.cs.toronto.edu/~kriz/cifar.html)
* CIFAR-10 数据集导入

```python
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

`x_train`：训练集图像，形状为 (50000, 32, 32, 3)

`y_train`：训练集标签，取值范围为 0-9 ，形状为 (50000, 1)

`x_test`：测试集图像，形状为 (10000, 32, 32, 3)

`y_test`：测试集标签，取值范围为 0-9 ，形状为 (10000, 1)

# 4 网络模型搭建、训练、测试与评估
虽然五种网络模型在结构和应用表现上各不相同，但它们拥有相似的搭建、训练、测试和评估流程，在 **Tensorflow 的 Keras 框架**下，整个过程主要分为以下五步：
* Step1. 定义网络类，搭建模型各层结构，初始化模型
```python
class ModelName(Model):
    def __init__(self):
        super(ModelName, self).__init__()
        ...
        描述各层结构
        ...

    def call(self, x):
        ...
        定义前向传播
        ...
        return y

model = ModelName()
```

* Step2. 设定优化器、损失函数与衡量指标
```python
model.compile(optimizer = <优化器>,
              loss = <损失函数>,
              metrics = <衡量指标>)
```

* Step3. 设定批大小、训练轮数等参数，训练网络并测试验证
```python
history = model.fit(x_train, y_train, 
                    batch_size = <批大小>,
                    epochs = <训练轮数>, 
                    validation_data = (x_test, y_test),
                    validation_freq = 1)
```
值得一提的是，对于训练参数过多、训练时间久的模型，或是训练可能被中断，可通过回调函数实现**断点续训**功能，保存当前训练结果，在下次训练时直接读取并继续训练。
```python
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(...
                    其他参数
                    ...
                    callbacks = [cp_callback])

```

* Step4. 保存训练好的模型到本地
```python
model.save('my_model/my_ModelName')
```

* Step5. 绘制训练集与测试集的 Accuracy 与 Loss 曲线
```python
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
...
使用 matplotlib 库绘图
...
```

## 4.1 LeNet-5
* 网络介绍

LeNet-5 是卷积神经网络的祖师爷 LeCun 在 1998 年提出的，用于解决手写数字识别的视觉任务，它是卷积神经网络早期最有代表性的网络之一。LeNet-5 网络共有 **5** 层，其中包括 **3** 层卷积层，**2** 层全连接层。您可以运行`LeNet5.py`训练该网络。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/LeNet-5_architecture.jpg" width=500></div>

* 训练结果

从变化趋势上看，训练集与测试集的 accuracy 在不断上升，loss 在不断下降；从分类效果上看，测试集 accuracy 只达到了 52% 左右，模型分类效果有待提升。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/LeNet5.jpg" width=500></div>

* 论文参考
  
  [LeCun, Yann, et al. “Gradient-based learning applied to document recognition.” Proceedings of the IEEE 86.11 (1998): 2278-2324.](https://ieeexplore.ieee.org/document/726791?reload=true&arnumber=726791)
  
## 4.2 AlexNet
* 网络介绍

AlexNet 是 2012 年 ImageNet 竞赛冠军获得者 Hinton 和他的学生 Alex Krizhevsky 设计的。它首次在卷积神经网络中成功应用了 ReLU、Dropout 和 LRN 等技巧。同时使用了 GPU 进行运算加速。AlexNet 网络共有 **8** 层，其中包括 **5** 层卷积层，**3** 层全连接层。您可以运行`AlexNet.py`训练该网络。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/AlexNet_architecture.jpg" width=500></div>

* 训练结果

从变化趋势上看，训练集表现正常，其 accuracy 在不断上升，loss 在不断下降，而测试集出现了比较大的震荡，可能需要对学习率和批大小等超参数进行调整，但整体趋势与训练集表现相同；从分类效果上看，测试集 accuracy 达到了 66% 左右，模型分类效果仍有待提升。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/AlexNet.jpg" width=500></div>


* 论文参考
  
  [Technicolor T , Related S , Technicolor T , et al. ImageNet Classification with Deep Convolutional Neural Networks.](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
## 4.3 VGGNet
* 网络介绍

VGGNet 是由 Simonyan 和 Zisserman 在 2014 年提出的模型。它反复堆叠 3 × 3 的小型卷积核和 2 × 2 的池化核，形成 5 段卷积，使得模型在架构上更深更宽的同时，计算量的增加规模放缓。VGGNet-16 网络共有 **16** 层，其中包括 **13** 层卷积层，**3** 层全连接层。您可以运行`VGGNet.py`训练该网络。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/VGG16_architecture.jpg" width=500></div>

* 训练结果

从变化趋势上看，训练集表现正常，其 accuracy 在不断上升，loss 在不断下降，而测试集出现了好几次震荡，但整体趋势与训练集表现相同，在训练后期，accuracy 有下降趋势且loss有上升趋势，模型即将过拟合；从分类效果上看，测试集 accuracy 达到了 79% 左右，较前两个模型有显著提升。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/VGGNet.jpg" width=500></div>

* 论文参考
  
  [Simonyan K , Zisserman A . Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](https://arxiv.org/pdf/1409.1556.pdf)

## 4.4 GoogLeNet
* 网络介绍

GoogLeNet 是 2014年 Christian Szegedy 设计的一种全新的网络结构，它提出了 inception 模块的概念，模块使用了 **1 x 1** 的卷积进行升降维，并在多个尺寸上同时进行卷积再聚合。
<div align=center><img src="https://github.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/blob/master/image/screenshot_image/Inception%20module_architecture.jpg" width=500></div>

GoogLeNet 由九个这样的 inception 模块串联起来，每个模块有 **2** 层，加上开头的 **3** 层卷积层和输出前的 **1** 层全连接层，整个网络共有 **22** 层。您可以运行`GoogLeNet.py`训练该网络。

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/GoogLeNet_architecture.jpg" width=500></div>

* 训练结果

从变化趋势上看，训练集与测试集的 accuracy 在不断上升，loss 在不断下降，测试集上的loss 下降速率逐渐趋于 0 ；从分类效果上看，测试集 accuracy 达到了 73% 左右，对于此模型来说仍然有待优化的空间，您可以调整相关超参数以提高模型分类效果。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/GoogLeNet.jpg" width=500></div>

* 论文参考
  
  [Szegedy C , Liu W , Jia Y , et al. Going Deeper with Convolutions[J]. IEEE Computer Society, 2014.](https://arxiv.org/pdf/1409.4842.pdf)

## 4.5 ResNet
* 网络介绍

ResNet 是由来自 Microsoft Research 的 4 位学者于 2015 年提出的卷积神经网络，它提出了残差单元的概念，其使用跳跃连接，将浅层网络的输出加给深层网络的输出，缓解了在深度神经网络中增加深度带来的梯度消失问题。残差网络的特点是容易优化，并且能够通过增加相当的深度来提高准确率。您可以运行`ResNet.py`训练该网络。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/Residual_block_architecture.jpg" width=500></div>


<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/ResNet18_architecture.jpg" width=500></div>

* 训练结果

从变化趋势上看，训练集 accuracy 在不断上升甚至达到 95%左右，loss 在不断下降，而测试集在训练前期的 accuracy 和 loss 就已经开始收敛，其后期 accuracy 有下降趋势且loss有上升趋势，模型表现为过拟合；从分类效果上看，测试集 accuracy 达到了 80% 左右，但由于模型过拟合，需要重新调整再训练。
<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/ResNet.jpg" width=500></div>

* 论文参考
  
  [He K , Zhang X , Ren S , et al. Deep Residual Learning for Image Recognition[J]. IEEE, 2016.](https://arxiv.org/pdf/1512.03385.pdf)


# 5 CIFAR-10 图像分类软件
## 5.1 使用说明
* Step1. 导入图片
  
  运行`main.py`打开软件。点击 <kbd><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/icons/open.png" width=30></kbd> 按钮打开一张图片，图片显示在左边区域
* Step2. 选择模型
  
  右上区域有五种分类模型供选择，默认选择为 LeNet-5
* Step3. 分类预测
  
  点击 <kbd><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/icons/predict.png" width=30></kbd> 按钮显示预测结果，结果输出 10 个类别中概率最大的类别

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/software_1.jpg" width=500></div>

<div align=center><img src="https://github.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/blob/master/image/screenshot_image/software_2.jpg" width=500></div>

## 5.2 测试数据
项目提供了用于软件测试的数据，它们由 CIFAR-10 数据集的 10 个类别组成，每个类别各有 5 张图。在选取制作这些图像时，遵从了同一类别中颜色、视角、背景、图像大小等存在一定差异的规则，避免图像相似度过高。

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/test_images_show.jpg" width=500></div>

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/test_images_airplane_show.jpg" width=500></div>

## 5.3 常见问题
* Q：分类预测时软件卡顿，等待时间过长
  
  A：由于部分网络模型层数和参数较多，使用加载时需要一定时间，请耐心等待
* Q：分类结果显示与现实不符
  
  A：由于训练的模型采用默认统一的参数，不同模型网络结构不同，图像分类的准确率存在差异，且无法达到 100%，您可以结合下文中超参数的研究自行调整参数训练模型，提高分类准确率

# 6 超参数研究
本节对六种常见的超参数进行了单一变量的研究，您可以尝试更多不同的值或是更多其他的超参数对卷积神经网络进行深入研究，以得到更多有价值的发现以及更好的模型分类效果。
## 6.1 学习率 

改变学习率大小，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.0001、0.0003、0.001、0.003、0.01
 optimizer |  Adam
 loss function |  SparseCategoricalCrossentropy
 activation function |  Relu
 batch-size | 32 
 epoch | 10 


* 模型表现

learning rate | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 0.0001 | 0.5496 | 0.5476 | 1.2334 | 1.2509 | 257s |
 0.0003 | 0.6894 | 0.6855 | 0.8766 | 0.8889 | 259s |
 0.001 | 0.7893 | 0.7210 | 0.5949 | 0.8404 | 259s |
 0.003 | 0.7562 | 0.7040 | 0.6886 | 0.8454 | 258s |
 0.01 | 0.1013 | 0.1000 | 2.3039 | 2.3039 | 258s |
  
<div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20learning%20rate.jpg" width=500></div>

* 结果分析

学习率从 0.0001 增长到 0.001 的过程中，模型分类的准确度在不断提升；后来增加到 0.003 时，效果反而下降；甚至学习率取 0.1 时，曲线近乎成了一条直线。

初始学习率需要选定一个恰当的值，并不是越大或者越小就越好。如果过低，可能导致训练过慢，需要很久才能到达很好的效果；如果过高，将很可能陷入局部极小值，无法跳出。

## 6.2 优化器
改变优化器类型，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.001
 optimizer |  SGD、RMSprop、AdaGrad、AdaDelta、Adam
 loss function |  SparseCategoricalCrossentropy
 activation function |  Relu
 batch-size | 32 
 epoch | 10 


* 模型表现

optimizer | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 SGD | 0.3352 | 0.3143 | 1.8473 | 1.8932 | 252s |
 RMSprop | 0.7721 | 0.7126 | 0.6516 | 0.8594 | 291s |
 AdaGrad | 0.2974 | 0.2905 | 1.9084 | 1.9276 | 263s |
 AdaDelta | 0.1131 | 0.1092 | 2.2927 | 2.2936 | 265s |
 Adam | 0.8021 | 0.7568 | 0.5568 | 0.7360 | 269s |
  
<div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20optimizer.jpg" width=500></div>

* 结果分析

从图中可以看到，选择 Adam 与 RMSprop 作为优化器的模型效果要远好于其他三种。但这往往不是绝对的，由于只研究单一变量，固定的初始学习率 0.001 可能并不适用于部分优化器，但从整体和大量实践经验来看，Adam 确实是收敛速度和效果都很不错的选择。

## 6.3 损失函数
改变损失函数类型，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.001
 optimizer | Adam
 loss function |  Mean Squared Error、Binary CrossEntropy、Sparse Categorical CrossEntropy
 activation function |  Relu
 batch-size | 32 
 epoch | 10 


* 模型表现

loss function | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 Mean Squared Error | 0.1002 | 0.1042 | 27.6100 | 27.6100 | 269s |
 Binary CrossEntropy | 0.0997 | 0.1000 | 9.9929 | 9.9929 | 261s |
 Sparse Categorical CrossEntropy | 0.7902 | 0.7252 | 0.5898 | 0.8163 | 255s |
 
 <div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20loss%20function.jpg" width=500></div>
 
 * 结果分析
 
从图中可以看到，均方误差 Mean Squared Error 和 二分类交叉熵 Binary CrossEntropy 损失函数的模型效果远不如 多分类交叉熵损失函数 Sparse Categorical CrossEntropy. 这是由于本项目是一个图像多分类问题，而前两者分别常用于回归问题和二分类问题中，所以根据机器学习具体应用场景选择对应的损失函数很重要，否则效果差距将非常大。
 
## 6.4 激活函数
改变激活函数类型，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.001
 optimizer | Adam
 loss function |  SparseCategoricalCrossentropy
 activation function |  Sigmoid、Tanh、Relu
 batch-size | 32 
 epoch | 10 


* 模型表现

activation function | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 Sigmoid | 0.5282 | 0.5184 | 1.2995 | 1.2998 | 274s |
 Tanh | 0.7803 | 0.7166 | 0.6294 | 0.8125 | 271s |
 Relu | 0.7887 | 0.7300 | 0.5965 | 0.7784 | 266s |
 
 <div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20activation%20function.jpg" width=500></div>
 
 * 结果分析

从结果可以看出，三种激活函数在本项目的效果上排序为：ReLu > Tanh > Sigmoid. 

Sigmoid 是最早的传统激活函数之一，它的缺点是很容易使梯度消失；Tanh 具有中心对称性，但也面临着梯度消失的风险，但它的性能和收敛速度比 Sigmoid 要略胜一筹；Relu 是深度学习实际应用中最广泛的激活函数，由于其在 x 大于 0 时导数恒为 1 ，所以不存在梯度消失的问题，收敛速度也远快于另两者。当然，现在已经有很多对 Relu 函数的改进和变形，如Leaky Relu、ELU等，  Relu 存在的一些缺点做了改进。
 
## 6.5 批尺寸
改变批尺寸大小，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.001
 optimizer |  Adam
 loss function |  SparseCategoricalCrossentropy
 activation function |  Relu
 batch-size | 16、32、64、128、256 
 epoch | 10 


* 模型表现

batch-size | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 16 | 0.8072 | 0.7370 | 0.5468 | 0.7911 | 333s |
 32 | 0.7924 | 0.7500 | 0.5888 | 0.7418 | 257s |
 64 | 0.7642 | 0.6986 | 0.6726 | 0.8813 | 207s |
 128 | 0.7066 | 0.6595 | 0.8280 | 0.9547 | 177s |
 256 | 0.6510 | 0.6439 | 0.9878 | 0.9910 | 164s |

 <div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20batch-size.jpg" width=500></div>
 
 * 结果分析

从整体上看，批尺寸从 16 增长到 256 的过程中，模型的准确度在不断下降，损失在上升，训练时间很明显变少，但这并不意味着大的 batch-size 训练出来的模型效果就不好。
 
大的 batch-size 往往下降方向准确，曲线震荡小，图中效果不好的原因是训练时间还不够长，在同样的 epoch 下参数更新较少，需要更多的回合数训练，同时大的 batch-size 容易陷入局部最优；小的 batch-size 虽然在较少 epoch 内效果好，但引入的随机性较大，会导致曲线震荡较大，不利于模型收敛，因此批大小也需要慎重选择一个合适的值。

## 6.6 回合数
改变回合数大小，保持其他超参数不变，具体取值如下表：

hyper parameter | value
---------|----------
 learning rate| 0.001
 optimizer |  Adam
 loss function |  SparseCategoricalCrossentropy
 activation function |  Relu
 batch-size | 32
 epoch | 5、10、20、50


* 模型表现

epoch | tra_acc | val_acc | tra_loss | val_loss | time | 
---------|----------|---------|---------|---------|---------
 5 | 0.6803 | 0.6625 | 0.8992 | 0.9388 | 128s |
 10 | 0.7832 | 0.7286 | 0.6171 | 0.7774 | 256s |
 20 | 0.8887 | 0.7360 | 0.3069 | 0.9294 | 498s |
 50 | 0.9643 | 0.7357 | 0.1016 | 1.5622 | 1212s |

 <div><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/different%20epoch.jpg" width=500></div>
 
 * 结果分析

随着训练回合数增多，模型训练时间增长。从训练集看，准确率持续上升，损失持续下降；从验证集上看，一开始走向与训练集相同，但达到一定回合后（图中约为 15），准确率几乎不再提升甚至有下降趋势，而损失在增长。
 
这说明模型发生了过拟合的现象，即随着训练回合加深，虽然在训练集上表现很好，但在测试集上表现却不尽人意。所以在深度学习模型训练过程中，绘制 epoch 与 accuracy、loss 的关系曲线，有助于帮助我们及时了解模型的状态以采取必要的措施。

# 版权许可
整个项目在MIT许可下发布（详细请参阅 [许可证](https://github.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/blob/master/LICENSE) 文件）


# 联系作者
[![GitHub](https://img.shields.io/badge/GitHub-%E5%B0%8F%E6%98%8E%E5%90%8C%E5%AD%A6-blue)](https://github.com/ChenMingwei1999)

[![github](https://img.shields.io/badge/Email-xiaoming__cmw1999%40163.com-blue)](mailto:xiaoming__cmw1999%40163.com)


# 更新日志
:date: 2021-10-27：训练了五种卷积网络模型，开发了一款 CIFAR-10 图像分类软件

:date: 2021-11-09：通过调整 GoogLeNet 网络的各种超参数，研究其产生的影响
