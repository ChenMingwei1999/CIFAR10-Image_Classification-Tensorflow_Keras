import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras import Model
from matplotlib import pyplot as plt

# 设置输出样式
np.set_printoptions(threshold=np.inf)

# 导入CIFAR-10数据集
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据归一化，将图像像素转化为0-1的实数
x_train, x_test = x_train / 255.0, x_test / 255.0


# 模型定义
# CBA封装类定义
class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False)
        return x

# Inception块定义
class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x

# Inception网络定义
class Inception(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 新建模型
model = Inception(num_blocks=2, num_classes=10)

# 优化器、损失函数、评价指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 模型训练：（训练集x和y、每批样本数次、训练轮数、验证集、验证间隔次数）
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)

# 打印神经网络结构，模型参数输出
model.summary()

# 保存模型
model.save('my_model/my_GoogLeNet')

# 绘制accuracy与loss曲线
# 获取训练集和验证集的acc和loss
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epoch_num = len(acc)
# 画图
plt.subplot(1, 2, 1)
plt.xlabel('epoch')
plt.plot(range(1, epoch_num+1), acc, 'r', linewidth=2.0, label='Training Accuracy')
plt.plot(range(1, epoch_num+1), val_acc, 'b', linewidth=2.0, label='Validation Accuracy')
# 图表标题
plt.title('Training and Validation Accuracy\n[ GoogLeNet ]')
# 显示图例
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('epoch')
plt.plot(range(1, epoch_num+1), loss, 'r', linewidth=2.0, label='Training Loss')
plt.plot(range(1, epoch_num+1), val_loss, 'b', linewidth=2.0, label='Validation Loss')
# 图表标题
plt.title('Training and Validation Loss\n[ GoogLeNet ]')
# 显示图例
plt.legend()
# 子图之间的间距
plt.subplots_adjust(wspace=0.3)
plt.show()