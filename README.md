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
  - [6.4 batch-size](#64-batch-size)
  - [6.5 epoch](#65-epoch)
  - [6.6 激活函数](#66-激活函数)
  - [6.7 权重初始化](#67-权重初始化)
  - [6.8 正则化参数](#68-正则化参数)
- [版权许可](#版权许可)
- [联系作者](#联系作者)
- [更新日志](#更新日志)

<!-- /TOC -->

# 1 项目简介
## 1.1 主要工作
* 通过 Tensorflow 的 Keras 框架搭建了 LeNet-5、AlexNet、VGGNet 、GoogLeNet、ResNet 五种卷积网络模型
* 在 CIFAR-10 数据集上对上述五个网络进行训练、测试和评估，得到 Accuracy 和 Loss 曲线
* 使用训练得到的模型，通过 PyQt 开发了一款交互式图像分类软件
* 调整 ? 模型的学习率、优化器、损失函数等超参数，观察并研究其产生的影响

## 1.2 文件结构
```python
├── Readme.md                   // help
├── app                         // 应用
├── config                      // 配置
│   ├── default.json
│   ├── dev.json                // 开发环境
│   ├── experiment.json         // 实验
│   ├── index.js                // 配置控制
│   ├── local.json              // 本地
│   ├── production.json         // 生产环境
│   └── test.json               // 测试环境
├── data
├── doc                         // 文档
├── environment
├── gulpfile.js
├── locales
├── logger-service.js           // 启动日志配置
├── node_modules
├── package.json
├── app-service.js              // 启动应用配置
├── static                      // web静态资源加载
│   └── initjson
│       └── config.js         // 提供给前端的配置
├── test
├── test-service.js
└── tools
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
CIFAR-10 数据集是一个用于识别普适物体的小型数据集，它包含 10 个类别、60000 张大小为 32×32 的彩色 RGB 图像，每类各 6000 张图。其中测试集共 10000 张，单独构成一批，在每一类中随机取 1000 张单独组成；训练集由剩下的随机排列组成，共 50000 张，构成了 5 个训练批，每一批 10000 张图，值得注意的是，一个训练批中的各类图像的数量不一定相同。

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
虽然五种网络模型在结构和应用表现上各不相同，但它们拥有相似的搭建、训练、测试和评估流程，在 Tensorflow 的 Keras 框架下，整个过程主要分为以下五步：
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
值得一提的是，对于训练参数过多、训练时间久的模型，或是训练可能被中断，可通过回调函数实现断点续训功能，保存当前训练结果，在下次训练时直接读取并继续训练。
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

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/LeNet-5_architecture.jpg" width=500></div>

* 训练结果

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/LeNet5.jpg" width=500></div>

* 论文参考
  
  [LeCun, Yann, et al. “Gradient-based learning applied to document recognition.” Proceedings of the IEEE 86.11 (1998): 2278-2324.](https://ieeexplore.ieee.org/document/726791?reload=true&arnumber=726791)
  
## 4.2 AlexNet
* 网络介绍

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/AlexNet_architecture.jpg" width=500></div>

* 训练结果

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/AlexNet.jpg" width=500></div>


* 论文参考
  
  [Technicolor T , Related S , Technicolor T , et al. ImageNet Classification with Deep Convolutional Neural Networks.](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
  
## 4.3 VGGNet
* 网络介绍

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/VGG16_architecture.jpg" width=500></div>

* 训练结果

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/VGGNet.jpg" width=500></div>

* 论文参考
  
  [Simonyan K , Zisserman A . Very Deep Convolutional Networks for Large-Scale Image Recognition[J]. Computer Science, 2014.](https://arxiv.org/pdf/1409.1556.pdf)

## 4.4 GoogLeNet
* 网络介绍

<div align=center><img src="https://github.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/blob/master/image/screenshot_image/Inception%20module_architecture.jpg" width=500></div>

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/GoogLeNet_architecture.jpg" width=500></div>

* 训练结果

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/GoogLeNet.jpg" width=500></div>

* 论文参考
  
  [Szegedy C , Liu W , Jia Y , et al. Going Deeper with Convolutions[J]. IEEE Computer Society, 2014.](https://arxiv.org/pdf/1409.4842.pdf)

## 4.5 ResNet
* 网络介绍

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/Residual_block_architecture.jpg" width=500></div>

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/screenshot_image/ResNet18_architecture.jpg" width=500></div>

* 训练结果

<div align=center><img src="https://raw.githubusercontent.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/master/image/evaluation%20metric_image/ResNet.jpg" width=500></div>

* 论文参考
  
  [He K , Zhang X , Ren S , et al. Deep Residual Learning for Image Recognition[J]. IEEE, 2016.](https://arxiv.org/pdf/1512.03385.pdf)


# 5 CIFAR-10 图像分类软件
## 5.1 使用说明
* Step1. 导入图片
  
  点击 <kbd></kbd> 按钮打开一张图片，图片显示在左边区域
* Step2. 选择模型
  
  右上区域有五种分类模型供选择，默认选择为 LeNet-5
* Step3. 分类预测
  
  点击 <kbd></kbd> 按钮显示预测结果，结果输出 10 个类别中概率最大的类别

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
  
  A：由于训练的模型采用默认统一的参数，不同模型网络结构不同，图像分类的准确率存在差异，且无法达到 100%，您可以结合下文中超参数的研究自行调整参数进行训练，提高分类准确率

# 6 超参数研究
## 6.1 学习率
## 6.2 优化器
## 6.3 损失函数
## 6.4 batch-size
## 6.5 epoch
## 6.6 激活函数
## 6.7 权重初始化
## 6.8 正则化参数


# 版权许可
整个项目在MIT许可下发布（详细请参阅 [许可证](https://github.com/ChenMingwei1999/CIFAR10-Image_Classification-Tensorflow_Keras/blob/master/LICENSE) 文件）


# 联系作者
[![GitHub](https://img.shields.io/badge/GitHub-%E5%B0%8F%E6%98%8E%E5%90%8C%E5%AD%A6-blue)](https://github.com/ChenMingwei1999)

[![github](https://img.shields.io/badge/Email-xiaoming__cmw1999%40163.com-blue)](mailto:xiaoming__cmw1999%40163.com)


# 更新日志
:date: 2021-10-27：训练了五种卷积网络模型，开发了一款 CIFAR-10 图像分类软件

:date: 2021-10-29：通过调整 ？ 各种超参数，研究其产生的影响
