# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
from keras.models import Sequential,model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from PIL import Image,ImageShow
import os.path, glob, cPickle


def change_size(img, width_new):
    (height, width) = img.size
    if height > width:
        offset = int(height - width) / 2
        img = img.transform((width, width), Image.EXTENT, (offset, 0, int(height - offset), width))
    else:
        offset = int(width - height) / 2
        img = img.transform((height, height), Image.EXTENT, (0, offset, height, (width - offset)))
    img.thumbnail((width_new, width_new))
    return img


def convert_data(input_dir, output_file, width, nb_classes):
    all_data = np.empty((0, 3, width, width), dtype="float32")
    all_label = np.empty((0,1), dtype="uint8")
    for k in range(nb_classes):
        imgs = os.listdir(input_dir+'/'+str(k))
        num = len(imgs)
        i = 0
        data = np.empty((num, 3, width, width), dtype="float32")
        label = np.empty((num,1), dtype="uint8")
        for file in glob.glob(input_dir+'/'+str(k) + '/*'):
            img = Image.open(file)
            img = change_size(img, width)
            # ImageShow.show(img)
            if img.size[0] < width or img.im.bands != 3:
                os.remove(file)
                print file, 'have been deleted!'
                continue
            array = np.asarray(img)
            for a in range(3):
                for b in range(width):
                    for c in range(width):
                        data[i, a, b, c] = array[b, c, a]
            # data[i, :, :, :] = array
            label[i] = k
            i += 1
        all_data = np.vstack((all_data, data))
        all_label = np.vstack((all_label, label))
    all_label = list(label[0] for label in all_label)
    data_label = {'data': all_data, 'label': np.array(all_label)}
    cPickle.dump(data_label, open(output_file, "wb"))


def load_data(input_file, nb_classes):
    data_label = cPickle.load(open(input_file, "rb"))
    data = data_label['data']
    label = data_label['label']
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    return x_train, x_test,  y_train, y_test


def config_model(model_str,width, nb_classes):
    # set model configuration
    model = Sequential()
    model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=(3, width, width)))
    model.add(Activation('tanh'))

    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, init='normal'))
    model.add(Activation('tanh'))

    model.add(Dense(nb_classes, init='normal'))
    model.add(Activation('softmax'))
    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


def save_model(model, model_str, model_weights):
    json_string = model.to_json()
    open(model_str, 'w').write(json_string)
    model.save_weights(model_weights, overwrite=True)


def load_model(model_str, model_weights, width, nb_classes):
    model = config_model(model_str, width, nb_classes)
    model.load_weights(model_weights)
    return model


def offline_learning(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train,
                batch_size=100, nb_epoch=10, verbose=1,
                validation_data=(x_test, y_test))
    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def online_learning(model, x_train, x_test, y_train, y_test):
    batch_size = 1
    nb_epoch = 3
    evaluate_checkpoint = 200
    iteration_checkpoint = 1000
    no_of_samples = len(x_train)
    # store (loss,accuracy) for train and test data
    train_score_list = []
    test_score_list = []
    for i in range(no_of_samples):
        # print info after consuming 1000 data points
        if ((i + 1) % iteration_checkpoint == 0):
            print("Example : %d", i)

        # train on ith data point
        model.fit(x_train[i:i + 1, :], y_train[i:i + 1],
                         batch_size, nb_epoch, verbose=0)

        # calculate regret on train and test
        if ((i + 1) % evaluate_checkpoint == 0) or i == 0:
            # compute scores and store them
            train_score = model.evaluate(x_train, y_train, verbose=0)
            test_score = model.evaluate(x_test, y_test, verbose=0)
            train_score_list.append(train_score)
            test_score_list.append(test_score)
            print 'train_score', train_score
            print 'test_score', test_score
    return model


def model_train(data_file, model_str, model_weights, nb_classes, width):
    x_train, x_test, y_train, y_test = load_data(data_file, nb_classes)
    model = load_model(model_str, model_weights, width, nb_classes)
    # model = online_learning(model, x_train, x_test, y_train, y_test)
    model = offline_learning(model, x_train, x_test, y_train, y_test)
    save_model(model, model_str, model_weights)


def model_predict(file_name, model_str, model_weights, width):
    # Test perform in test dataSet
    model = load_model(model_str, model_weights, width, nb_classes)
    img = Image.open(file_name)
    img = change_size(img, width)
    if img.size[0] < width:
        print file, ' size is not correct!'
    data = np.empty((1, 3, width, width), dtype="float32")
    array = np.asarray(img)
    for a in range(3):
        for b in range(width):
            for c in range(width):
                data[0, a, b, c] = array[b, c, a]
    predicted_classes = model.predict_classes(data)
    return data, predicted_classes


def feedback_train(data, predicted_classes, is_true, model_str, model_weights, width):
    label = np.empty((1, 1), dtype="uint8")
    sample_weight = np.empty((1, ), dtype="uint8")
    if is_true == 0:
        label[0] = 1-predicted_classes
        sample_weight[0] = 10
    model = load_model(model_str, model_weights, width, nb_classes)
    label = np_utils.to_categorical(label, nb_classes)
    model.fit(data, label, 1, 3, verbose=0, sample_weight=sample_weight)
    save_model(model, model_str, model_weights)


if __name__ == '__main__':
    width = 50
    nb_classes = 6
    origin_dir = 'dataset/'
    train_data = 'dataset/data_6.pkl'
    model_str = 'model/model.json'
    model_weights = 'model/weights_6.h5'
    test_file = 'dataset/0/n04971313_31.JPEG'
    test_label = 0
    # for i in range(2): convert_data(origin_dir, train_data, width, nb_classes)
    # model_train(train_data, model_str, model_weights, nb_classes, width)

    data, predicted_classes = model_predict(test_file, model_str, model_weights, width)
    print predicted_classes
    feedback_train(data, predicted_classes, 1, model_str, model_weights, width)
