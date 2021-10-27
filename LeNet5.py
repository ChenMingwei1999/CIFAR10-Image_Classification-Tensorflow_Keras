import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
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
class LeNet5(Model):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.c2 = Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


# 新建模型
model = LeNet5()

# 优化器、损失函数、评价指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 模型训练：（训练集x和y、每批样本数次、训练轮数、验证集、验证间隔次数）
history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), validation_freq=1)

# 打印神经网络结构，模型参数输出
model.summary()

# 保存模型
model.save('my_model/my_LeNet')

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
plt.title('Training and Validation Accuracy\n[ LeNet5 ]')
# 显示图例
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('epoch')
plt.plot(range(1, epoch_num+1), loss, 'r', linewidth=2.0, label='Training Loss')
plt.plot(range(1, epoch_num+1), val_loss, 'b', linewidth=2.0, label='Validation Loss')
# 图表标题
plt.title('Training and Validation Loss\n[ LeNet5 ]')
# 显示图例
plt.legend()
# 子图之间的间距
plt.subplots_adjust(wspace=0.3)
plt.show()