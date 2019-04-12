import urllib.request
import os
import numpy as np
import zipfile
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
# import cv2 as cv
import re
import keras
from keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
# Just disables the warning, doesn't enable AVX/FMA
import os
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 3


def get_label_from_name(image_name):
    result = ''
    p = re.compile('Class_([A-C])_\d.*')
    m = p.match(image_name)
    if (m != None):
        result = m.group(1)

    return result


def display_image(im):
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def convert_to_one_hot_encoding(labels_arr, first_class, num_classes):
    result_y = []
    for label in labels_arr:
        result_label = np.zeros(num_classes)
        result_label[ord(label) - ord(first_class)] = 1
        result_y.append(result_label)

    return result_y


def load_train_data():
    data_path = 'C:/repo/HackIDC data/Data/'
    train_data_directory = os.listdir(data_path + 'trainData')
    images = []
    labels = []

    for directory in train_data_directory:
        files_list = os.listdir(data_path + 'trainData/' + directory)
        for f in files_list:
            file_path = data_path + 'trainData/' + directory + '/' + f
            if f.endswith('.jpg'):
                img_label = get_label_from_name(f)
                labels.append(img_label)
                img = np.asarray(Image.open(file_path))
                images.append(img)

    return images, labels


def load_test_data():
    data_path = 'C:/repo/HackIDC data/Data/'
    test_data_directory = os.listdir(data_path + 'testData')
    test_images = []
    test_labels = []
    test_files_list = os.listdir(data_path + 'testData')

    for f in test_files_list:
        file_path = data_path + 'testData/' + f
        if f.endswith('.jpg') or f.endswith('.jpeg'):
            img_label = get_label_from_name(f)
            test_labels.append(img_label)
            img = np.asarray(Image.open(file_path))
            test_images.append(img)

    return test_images, test_labels


def set_model_layers(input_model):
    input_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    input_model.add(keras.layers.BatchNormalization())

    input_model.add(keras.layers.Conv2D(32, (3, 3), input_shape=(240, 320, 3)))
    input_model.add(keras.layers.Activation('relu'))
    input_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    input_model.add(keras.layers.BatchNormalization())

    input_model.add(keras.layers.Flatten())
    input_model.add(keras.layers.Dense(512))
    input_model.add(keras.layers.Activation('relu'))
    input_model.add(keras.layers.BatchNormalization())
    input_model.add(keras.layers.Dense(num_classes))
    input_model.add(keras.layers.Activation('softmax'))


def load_data_and_set_model_layers(input_model):
    train_images, train_labels = load_train_data()
    test_images, test_labels = load_test_data()
    print('after images loading')
    train_images = np.array(train_images).astype('float32') / 255
    test_images = np.array(test_images).astype('float32') / 255

    train_labels = convert_to_one_hot_encoding(train_labels, 'A', num_classes)
    test_labels = convert_to_one_hot_encoding(test_labels, 'A', num_classes)

    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images,
                                                                                        train_labels, test_size=0.2)
    train_images = np.array(train_images)
    validation_images = np.array(validation_images)
    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    set_model_layers(input_model)

    return train_images, validation_images, train_labels, validation_labels, test_images, test_labels


def get_model(produce_new_model=False):

    if os.path.isfile('saved_model.h5') and not produce_new_model:
        model_in = load_model('saved_model.h5')
        print('Loading saved model')
    else:
        model_in = keras.models.Sequential()

        train_images, validation_images, train_labels, validation_labels, test_images, test_labels\
            = load_data_and_set_model_layers(model_in)

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model_in.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        model_in.fit(
            train_images, train_labels,
            batch_size=20,
            epochs=1,
            validation_data=(validation_images, validation_labels)
        )
        print('after new model compile and fit')
        model_in.save('saved_model.h5')

        loss, acc = model_in.evaluate(train_images, train_labels, verbose=1)
        print('Train loss:', loss)
        print('Train accuracy:', acc)
        loss, acc = model_in.evaluate(test_images, test_labels, verbose=1)
        print('Test loss:', loss)
        print('Test accuracy:', acc)

        predicted_labels_in = get_labels_from_class_numbers(model_in.predict_classes(test_images))
        print('Test results: ')
        print('Test labels: ')
        print(get_labels_array_from_one_hot_array(test_labels))
        print('Predicted labels: ')
        print(predicted_labels_in)

    return model_in


def get_classes_arr():
    classes_array = []
    for i in range(num_classes):
        classes_array.append(chr(ord('A') + i))

    return classes_array


def get_labels_from_class_numbers(i_class_numbers):
    labels_to_return = []
    for num in i_class_numbers:
        labels_to_return.append(chr(ord('A') + num))

    return labels_to_return


def get_label_from_one_hot(label_in_one_hot):
    classes_arr_in = get_classes_arr()
    return np.array(classes_arr_in)[label_in_one_hot == 1][0]


def get_labels_array_from_one_hot_array(labels_in_one_hot):
    result = []
    for label_in_one_hot in labels_in_one_hot:
        result.append(get_label_from_one_hot(label_in_one_hot))

    return result

'''
model = get_model()

test_images, test_labels = load_test_data()
test_images = np.array(test_images).astype('float32') / 255
print('test labels: ')
print(test_labels)
predicted_labels = get_labels_from_class_numbers(model.predict_classes([test_images]))
print('Test predicted classes: ')
print(predicted_labels)
'''