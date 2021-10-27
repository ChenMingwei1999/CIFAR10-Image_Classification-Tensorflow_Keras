import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, Activation
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
# 残差块定义
class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # 当residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x与F(x)维度相同，顺利相加
        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs

        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,加完后再过激活函数
        out = self.a2(y + residual)
        return out


# 残差网络定义
class ResNet(Model):
    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet, self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(self.num_blocks):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的残差块加入resnet
            self.out_filters *= 2  # 下一个残差块的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 新建模型
model = ResNet([2, 2, 2, 2])

# 优化器、损失函数、评价指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 模型训练：（训练集x和y、每批样本数次、训练轮数、验证集、验证间隔次数）
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)

# 打印神经网络结构，模型参数输出
model.summary()

# 保存模型
model.save('my_model/my_ResNet')

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
plt.title('Training and Validation Accuracy\n[ ResNet ]')
# 显示图例
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('epoch')
plt.plot(range(1, epoch_num+1), loss, 'r', linewidth=2.0, label='Training Loss')
plt.plot(range(1, epoch_num+1), val_loss, 'b', linewidth=2.0, label='Validation Loss')
# 图表标题
plt.title('Training and Validation Loss\n[ ResNet ]')
# 显示图例
plt.legend()
# 子图之间的间距
plt.subplots_adjust(wspace=0.3)
plt.show()