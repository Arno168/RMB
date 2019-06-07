from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model, load_model


class Train_Models:

    def cnn(self, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT):
        print("当前训练模型为CNN")
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, 3)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation="softmax"))
        return model

    def vgg16(self, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT):
        print("当前训练模型为VGG16")
        model_vgg = VGG16(include_top=False, weights="imagenet",
                          input_shape=[IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT, 3])
        # for layer in model_vgg.layers:
        #     #if layer_index < FREEZE_LAYER+1:
        #     layer.trainable = True

        model_self = Flatten(name='flatten')(model_vgg.output)
        # model_self = Dense(2048, activation='relu', name='fc2')(model_self)
        # model_self = Dropout(0.5)(model_self)
        model_self = Dense(9, activation='softmax')(model_self)
        model_vgg_9 = Model(inputs=model_vgg.input, outputs=model_self, name='vgg16')
        model_vgg_9.summary()
        return model_vgg_9
