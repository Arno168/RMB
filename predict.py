import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame,Series
from keras.models import Model, load_model
from keras.preprocessing.image import img_to_array

is_load_model = True
model_save_path = r'data\model_self_new.h5'
test_path = r'data\img\test_data\test_data'
IMAGE_SIZE_WIDTH = 64
IMAGE_SIZE_HEIGHT = 64

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

# one hot coding 转回字符串label
def vec2label(label_vec):
    mapping = {
        0: "0.1",
        1: "0.2",
        2: "0.5",
        3: "1",
        4: "2",
        5: "5",
        6: "10",
        7: "50",
        8: "100"
    }
    df = DataFrame(label_vec)
    ts = pd.Series(df[0].values)
    return ts.map(mapping)

# 取预测值的前k位
def get_top_k_label(preds, k=1):
    top_k = tf.nn.top_k(preds, k).indices
    with tf.Session() as sess:
        top_k = sess.run(top_k)
    top_k_label = vec2label(top_k)
    return top_k_label


if __name__ == '__main__':
    model = ""
    if is_load_model and os.path.exists(model_save_path):
        model = load_model(model_save_path)

    files = os.listdir(test_path)  # 列出文件夹下所有的目录与文件
    #files=files[0:5]
    print("测试集目录下文件数："+str(len(files)))

    test_set = load_image(test_path, files, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, 3)

    print("预测测试集开始")
    test_preds = model.predict(test_set)
    print("预测测试集结束")

    print(test_preds)
    print(str(len(files)))
    predslabel = get_top_k_label(test_preds, 1)
    print(str(len(predslabel)))
    submit = pd.DataFrame({'name': files, 'label': predslabel})
    print("保存预测结果")
    submit.to_csv(r'data\submit.csv', index=False, line_terminator='\n')
    print("保存完毕")
    print("end")
