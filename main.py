import numpy as np
import sys
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
import tensorflow as tf
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from main_ui import *


# 软件界面类
class MyWindow(QWidget, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        # 无边框
        self.setWindowFlags(Qt.FramelessWindowHint)
        # 窗口大小不允许修改
        self.setFixedSize(self.width(), self.height())

        # 为控件加阴影
        self.shadow(self.exit_btn)
        self.shadow(self.minimize_btn)
        self.shadow(self.open_btn)
        self.shadow(self.predict_btn)
        self.shadow(self.pic_area)
        self.shadow(self.model_area)
        self.shadow(self.result_area)
        self.shadow(self.model_1_btn)
        self.shadow(self.model_2_btn)
        self.shadow(self.model_3_btn)
        self.shadow(self.model_4_btn)
        self.shadow(self.model_5_btn)

        # 控件悬浮气泡提示
        QToolTip.setFont(QFont('微软雅黑', 8))  # 提示信息的字体与字号
        self.exit_btn.setToolTip("关闭软件")
        self.minimize_btn.setToolTip("最小化窗口")
        self.open_btn.setToolTip("打开图片")
        self.predict_btn.setToolTip("根据所选模型预测分类结果")

        # 软件关闭、最小化
        self.exit_btn.clicked.connect(self.close)
        self.minimize_btn.clicked.connect(self.showMinimized)

        # 打开图片
        self.open_btn.clicked.connect(self.open_pic)  # 打开图片
        self.img = ""

        # 模型按钮切换逻辑
        self.model_1_btn.clicked.connect(self.choose_model1)
        self.model_2_btn.clicked.connect(self.choose_model2)
        self.model_3_btn.clicked.connect(self.choose_model3)
        self.model_4_btn.clicked.connect(self.choose_model4)
        self.model_5_btn.clicked.connect(self.choose_model5)

        # 分类
        self.predict_btn.clicked.connect(self.classify)

    # 加阴影
    def shadow(self, item):
        shadow_effect = QGraphicsDropShadowEffect(self)
        shadow_effect.setOffset(0, 0)
        shadow_effect.setBlurRadius(10)
        item.setGraphicsEffect(shadow_effect)

    # 打开图片
    def open_pic(self):
        temp_filename, filetype = QFileDialog.getOpenFileName(self, "打开图片", "./", "(*.jpg);;(*.bmp);;(*.png)", "(*.jpg)")
        # opencv读入图片，三通道彩色图，默认为BGR格式,为了解决中文路径的问题，不能直接用imread
        if temp_filename:
            self.filename = temp_filename
            self.img = cv2.imdecode(np.fromfile(self.filename, dtype=np.uint8), -1)
            self.width = self.img.shape[1]
            self.height = self.img.shape[0]
            self.channel = self.img.shape[2]
            # 转换成RGB模式
            cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB, self.img)
            # 转成QImage是为了后面转成QPixmap形式放入控件显示
            qt_img = QtGui.QImage(self.img.data, self.width, self.height, self.width * self.channel,
                                  QtGui.QImage.Format_RGB888)
            # 为Graphics控件添加图片进行显示
            self.scene = QGraphicsScene()
            self.scene.addPixmap(QtGui.QPixmap.fromImage(qt_img))
            self.graphicsView.setScene(self.scene)
            self.graphicsView.show()
            # 导入新图后清除上次预测结果
            self.result_label.setText("？")

    # 分类模型切换逻辑
    def choose_model1(self):
        if self.model_1_btn.isChecked():
            self.model_1_btn.setChecked(True)
            self.model_2_btn.setChecked(False)
            self.model_3_btn.setChecked(False)
            self.model_4_btn.setChecked(False)
            self.model_5_btn.setChecked(False)
        else:
            self.model_1_btn.setChecked(True)

    def choose_model2(self):
        if self.model_2_btn.isChecked():
            self.model_2_btn.setChecked(True)
            self.model_1_btn.setChecked(False)
            self.model_3_btn.setChecked(False)
            self.model_4_btn.setChecked(False)
            self.model_5_btn.setChecked(False)
        else:
            self.model_2_btn.setChecked(True)

    def choose_model3(self):
        if self.model_3_btn.isChecked():
            self.model_3_btn.setChecked(True)
            self.model_1_btn.setChecked(False)
            self.model_2_btn.setChecked(False)
            self.model_4_btn.setChecked(False)
            self.model_5_btn.setChecked(False)
        else:
            self.model_3_btn.setChecked(True)

    def choose_model4(self):
        if self.model_4_btn.isChecked():
            self.model_4_btn.setChecked(True)
            self.model_1_btn.setChecked(False)
            self.model_2_btn.setChecked(False)
            self.model_3_btn.setChecked(False)
            self.model_5_btn.setChecked(False)
        else:
            self.model_4_btn.setChecked(True)

    def choose_model5(self):
        if self.model_5_btn.isChecked():
            self.model_5_btn.setChecked(True)
            self.model_1_btn.setChecked(False)
            self.model_2_btn.setChecked(False)
            self.model_3_btn.setChecked(False)
            self.model_4_btn.setChecked(False)
        else:
            self.model_5_btn.setChecked(True)

    def classify(self):
        # 加载模型
        if self.model_1_btn.isChecked():
            my_model = tf.keras.models.load_model('my_model/my_LeNet')
        if self.model_2_btn.isChecked():
            my_model = tf.keras.models.load_model('my_model/my_AlexNet')
        if self.model_3_btn.isChecked():
            my_model = tf.keras.models.load_model('my_model/my_VGGNet')
        if self.model_4_btn.isChecked():
            my_model = tf.keras.models.load_model('my_model/my_GoogLeNet')
        if self.model_5_btn.isChecked():
            my_model = tf.keras.models.load_model('my_model/my_ResNet')

        # 测试图片加载
        if self.img != "":
            test_images = self.img
            test_images = cv2.resize(test_images, (32, 32))

            # 归一化处理
            test_images = test_images / 255.0
            # 把图片转化为4维
            test_images = test_images.reshape(1, 32, 32, 3)

            # 用模型对待测图片进行分类预测，result为各分类的概率
            result = my_model.predict(test_images)

            # 分类序号
            max_index = np.argmax(result, axis=1)
            max_index = max_index[0]

            if max_index == 0:
                result_class = '飞机'
            elif max_index == 1:
                result_class = '汽车'
            elif max_index == 2:
                result_class = '鸟'
            elif max_index == 3:
                result_class = '猫'
            elif max_index == 4:
                result_class = '鹿'
            elif max_index == 5:
                result_class = '狗'
            elif max_index == 6:
                result_class = '蛙'
            elif max_index == 7:
                result_class = '马'
            elif max_index == 8:
                result_class = '船'
            elif max_index == 9:
                result_class = '卡车'

            self.result_label.setText(result_class)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())

