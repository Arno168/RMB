
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import img_to_array
from keras.optimizers import Adam
from keras import regularizers
from model import Train_Models

is_load_model = False
model_save_path = r'data\model_vgg16.h5'
train_path = r'data\img\train_data\train_data'
test_path = r'data\img\test_data\test_data'
train_label_path = r'data\train_face_value_label.csv'

EPOCHS_SIZE = 30  # 30
BATCH_SIZE = 32
INIT_LR = 1e-3  # 0.01
DECAY = 1e-5
IMAGE_SIZE_WIDTH = 64
IMAGE_SIZE_HEIGHT = 64


# 划分训练集和验证集
def make_train_and_val_set(dataset, labels, test_size):
    train_set, val_set, train_label, val_label = train_test_split(dataset, labels,
                                                                  test_size=test_size, random_state=5)
    return train_set, val_set, train_label, val_label

# 图片读取
def load_image(path, filesname, height, width, channels):
    images = []
    for image_name in filesname:
        image = cv2.imread(os.path.join(path, image_name))
        image = cv2.resize(image, (height, width))
        images.append(img_to_array(image))
        print("已加载:"+str(len(images))+"张图片")
    images = np.array(images, dtype="float") / 255.0
    images = images.reshape([-1, height, width, channels])
    #images = np.expand_dims(images, axis=0)
    print(images.shape)
    return images




# 数据增强
train_datagen = ImageDataGenerator(
    horizontal_flip=True
 )

if __name__ == '__main__':
    train_data = pd.read_csv(train_label_path)
    train_data = train_data[0:100]
    train_filesname = train_data["name"]
    train_labels = train_data[" label"]

    # 输出面额
    face_values = np.sort(train_data[" label"].unique())
    print(face_values)

    # 查看每种货币的样本数量是否倾斜
    sample = pd.read_csv(train_label_path, names=['name', ' label'], skiprows=1)
    sample['label_cat'] = sample[" label"].astype('category').cat.codes.astype(int)
    sample[" label"] = sample[" label"].astype('str')
    sample.head()
    print(sample.label_cat.value_counts().sort_index())

    # 独热编码
    train_labels = pd.get_dummies(train_labels)

    print("划分训练集和验证集")
    train_set_name, val_set_name, train_label, val_label = make_train_and_val_set(train_filesname, train_labels, 0.2)

    print("加载训练集图片")
    train_set = load_image(train_path, train_set_name, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, 3)

    print("加载验证集图片")
    val_set = load_image(train_path, val_set_name, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, 3)

    print("创建模型")

    # 载入模型
    if is_load_model and os.path.exists(model_save_path):
        print("加载模型")
        model = load_model(model_save_path)
        #opt = SGD(lr=INIT_LR, decay=DECAY)
    else:
        models = Train_Models()
        model = models.vgg16(IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT)
    from keras.callbacks import ReduceLROnPlateau
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    # train_set.shape[0] # batch_size
    steps_per_epoch = len(train_set) // BATCH_SIZE
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("训练开始")
    history = model.fit_generator(train_datagen.flow(train_set, train_label, batch_size=BATCH_SIZE),
                                  epochs=EPOCHS_SIZE, validation_data=(val_set, val_label),
                                  verbose=-1, steps_per_epoch=train_set.shape[0],
                                  callbacks=[learning_rate_reduction])

    print(history.history['loss'])
    print(history.history['acc'])
    print("训练结束")
    print("绘制损失函数图像")

    # 保存模型
    print("保存模型开始")
    model.save(model_save_path)
    print("保存模型结束")

    print("end")

