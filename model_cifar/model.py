from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten
from tensorflow.keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization
#import keras_resnet.models
import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50     import ResNet50
from tensorflow.keras.applications.vgg16        import VGG16
from tensorflow.keras.applications.inception_v3        import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
import keras_resnet.models  #https://github.com/broadinstitute/keras-resnet

import classifiers.resnet_jetett
import tensorflow as tf

my_activation = 'relu'

FINAL_DENSE_NUM = 512


def create_model_alexnet(in_shape,nb_classes):
    # http://www.michaelfxu.com/neural%20networks%20series/neural-networks-pt4-cnn-codes/
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=in_shape, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=my_activation))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=my_activation))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=my_activation))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation=my_activation))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation=my_activation))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_model_alexnet_act(in_shape,nb_classes):
    # http://www.michaelfxu.com/neural%20networks%20series/neural-networks-pt4-cnn-codes/
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=in_shape, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh')) #, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation=my_activation))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_model_alexnet_random(in_shape,nb_classes):
    # http://www.michaelfxu.com/neural%20networks%20series/neural-networks-pt4-cnn-codes/
    model = Sequential()
    model.add(Conv2D(filters=32, input_shape=in_shape, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.))) #, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(512, activation=my_activation))
    model.add(Dropout(0.3))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_RaptorMai(in_shape,nb_classes):
    # https://github.com/RaptorMai/Cifar10-CNN-Keras/blob/master/Cifar_better.ipynb
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=in_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_RaptorMai_zero(in_shape,nb_classes):
    # https://github.com/RaptorMai/Cifar10-CNN-Keras/blob/master/Cifar_better.ipynb
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros(),
                     input_shape=in_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.Zeros()))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_RaptorMai_random(in_shape,nb_classes):
    # https://github.com/RaptorMai/Cifar10-CNN-Keras/blob/master/Cifar_better.ipynb
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.),
                     input_shape=in_shape))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_RaptorMai_act1(in_shape,nb_classes):
    # https://github.com/RaptorMai/Cifar10-CNN-Keras/blob/master/Cifar_better.ipynb
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=in_shape))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def create_RaptorMai_act2(in_shape,nb_classes):
    # https://github.com/RaptorMai/Cifar10-CNN-Keras/blob/master/Cifar_better.ipynb
    weight_decay = 1e-4
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                     input_shape=in_shape))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('tanh'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))
    return model

def conv2d_bn_relu(x, filters, kernel_size, name, weight_decay=.0, strides=(1, 1), use_bn=True):
    conv_name = name + "-conv"
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               kernel_regularizer=regularizers.l2(weight_decay),
               name=conv_name)(x)
    if use_bn:
        bn_name = name + "-bn"
        x = BatchNormalization(scale=False, axis=3, name=bn_name)(x)
    relu_name = name + "-relu"
    x = Activation('relu', name=relu_name)(x)

    return x

def conv2d_bn_tanh(x, filters, kernel_size, name, weight_decay=.0, strides=(1, 1), use_bn=True):
    conv_name = name + "-conv"
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               kernel_regularizer=regularizers.l2(weight_decay),
               name=conv_name)(x)
    if use_bn:
        bn_name = name + "-bn"
        x = BatchNormalization(scale=False, axis=3, name=bn_name)(x)
    tanh_name = name + "-tanh"
    x = Activation('tanh', name=tanh_name)(x)

    return x

def inception_block_v1_act(x, filters_num_array, name, weight_decay=.0, use_bn=True):
    """

    :param x: model
    :param filters_num_array: filters num is 4 branch format (1x1, (1x1, 3x3), (1x1, 5x5), (pool, 1x1))
    :return: block added model x
    """
    (br0_filters, br1_filters, br2_filters, br3_filters) = filters_num_array
    # br0
    br0 = conv2d_bn_tanh(x,
                         filters=br0_filters, kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br0-1x1', use_bn=use_bn)

    # br1
    br1 = conv2d_bn_tanh(x,
                         filters=br1_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br1-1x1', use_bn=use_bn)
    br1 = conv2d_bn_tanh(br1,
                         filters=br1_filters[1], kernel_size=(3, 3), weight_decay=weight_decay,
                         name=name + '-br1-3x3', use_bn=use_bn)

    # br2
    br2 = conv2d_bn_tanh(x,
                         filters=br2_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br2-1x1', use_bn=use_bn)
    br2 = conv2d_bn_tanh(br2, filters=br2_filters[1], kernel_size=(5, 5), name=name + '-br2-5x5', use_bn=use_bn)

    # br3
    br3 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name + '-br3-pool')(x)
    br3 = conv2d_bn_tanh(br3, filters=br3_filters, kernel_size=(1, 1), weight_decay=weight_decay, name=name + '-br3-1x1')

    x = concatenate([br0, br1, br2, br3], axis=3, name=name)
    return x

def inception_block_v1(x, filters_num_array, name, weight_decay=.0, use_bn=True):
    """

    :param x: model
    :param filters_num_array: filters num is 4 branch format (1x1, (1x1, 3x3), (1x1, 5x5), (pool, 1x1))
    :return: block added model x
    """
    (br0_filters, br1_filters, br2_filters, br3_filters) = filters_num_array
    # br0
    br0 = conv2d_bn_relu(x,
                         filters=br0_filters, kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br0-1x1', use_bn=use_bn)

    # br1
    br1 = conv2d_bn_relu(x,
                         filters=br1_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br1-1x1', use_bn=use_bn)
    br1 = conv2d_bn_relu(br1,
                         filters=br1_filters[1], kernel_size=(3, 3), weight_decay=weight_decay,
                         name=name + '-br1-3x3', use_bn=use_bn)

    # br2
    br2 = conv2d_bn_relu(x,
                         filters=br2_filters[0], kernel_size=(1, 1), weight_decay=weight_decay,
                         name=name + '-br2-1x1', use_bn=use_bn)
    br2 = conv2d_bn_relu(br2, filters=br2_filters[1], kernel_size=(5, 5), name=name + '-br2-5x5', use_bn=use_bn)

    # br3
    br3 = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=name + '-br3-pool')(x)
    br3 = conv2d_bn_relu(br3, filters=br3_filters, kernel_size=(1, 1), weight_decay=weight_decay, name=name + '-br3-1x1')

    x = concatenate([br0, br1, br2, br3], axis=3, name=name)
    return x

def create_model_inceptionv1ForCifar10_act(in_shape,nb_classes):
    weight_decay= 5e-4
    use_bn=True
    input = Input(shape=in_shape)
    x = input
    # x = conv2d_bn_relu(x,
    #                    filters=64, kernel_size=(7, 7), name='1a',
    #                    weight_decay=weight_decay, strides=(2, 2), use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='1-pool')(x)
    x = conv2d_bn_tanh(x,
                       filters=64, kernel_size=(1, 1), weight_decay=weight_decay,
                       name='2a', use_bn=use_bn)
    x = conv2d_bn_tanh(x, filters=192, kernel_size=(3, 3), weight_decay=weight_decay,
                       name='2b', use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='2-pool')(x)

    # inception3a
    x = inception_block_v1_act(x, (64, (96, 128), (16, 32), 32),
                           weight_decay=weight_decay,
                           name='inception3a', use_bn=use_bn)
    # inception3b
    x = inception_block_v1_act(x, (128, (128, 192), (32, 96), 64),
                           weight_decay=weight_decay,
                           name='inception3b', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='3pool')(x)

    # inception4a
    x = inception_block_v1_act(x, (192, (96, 208), (16, 48), 64),
                           weight_decay=weight_decay,
                           name='inception4a', use_bn=use_bn)
    # inception4b
    x = inception_block_v1_act(x, (160, (112, 224), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4b', use_bn=True)
    # inception4c
    x = inception_block_v1_act(x, (128, (128, 256), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4c', use_bn=use_bn)
    # inception4d
    x = inception_block_v1_act(x, (112, (144, 288), (32, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4d', use_bn=use_bn)
    # inception4e
    x = inception_block_v1_act(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception4e', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='4pool')(x)

    # inception5a
    x = inception_block_v1_act(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5a', use_bn=use_bn)
    # inception5b
    x = inception_block_v1_act(x, (384, (192, 384), (48, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5b', use_bn=use_bn)

    # average pool
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid', name='avg8x8')(x)
    # x = Dropout(0.4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    model = Model(input, x, name='inception_v1')
    return model

def create_model_inceptionv1ForCifar10_jerett(in_shape,nb_classes):
    weight_decay= 5e-4
    use_bn=True
    input = Input(shape=in_shape)
    x = input
    # x = conv2d_bn_relu(x,
    #                    filters=64, kernel_size=(7, 7), name='1a',
    #                    weight_decay=weight_decay, strides=(2, 2), use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='1-pool')(x)
    x = conv2d_bn_relu(x,
                       filters=64, kernel_size=(1, 1), weight_decay=weight_decay,
                       name='2a', use_bn=use_bn)
    x = conv2d_bn_relu(x, filters=192, kernel_size=(3, 3), weight_decay=weight_decay,
                       name='2b', use_bn=use_bn)
    # x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='2-pool')(x)

    # inception3a
    x = inception_block_v1(x, (64, (96, 128), (16, 32), 32),
                           weight_decay=weight_decay,
                           name='inception3a', use_bn=use_bn)
    # inception3b
    x = inception_block_v1(x, (128, (128, 192), (32, 96), 64),
                           weight_decay=weight_decay,
                           name='inception3b', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='3pool')(x)

    # inception4a
    x = inception_block_v1(x, (192, (96, 208), (16, 48), 64),
                           weight_decay=weight_decay,
                           name='inception4a', use_bn=use_bn)
    # inception4b
    x = inception_block_v1(x, (160, (112, 224), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4b', use_bn=True)
    # inception4c
    x = inception_block_v1(x, (128, (128, 256), (24, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4c', use_bn=use_bn)
    # inception4d
    x = inception_block_v1(x, (112, (144, 288), (32, 64), 64),
                           weight_decay=weight_decay,
                           name='inception4d', use_bn=use_bn)
    # inception4e
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception4e', use_bn=use_bn)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='4pool')(x)

    # inception5a
    x = inception_block_v1(x, (256, (160, 320), (32, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5a', use_bn=use_bn)
    # inception5b
    x = inception_block_v1(x, (384, (192, 384), (48, 128), 128),
                           weight_decay=weight_decay,
                           name='inception5b', use_bn=use_bn)

    # average pool
    x = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='valid', name='avg8x8')(x)
    # x = Dropout(0.4)(x)
    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, activation='softmax', name='predictions')(x)
    model = Model(input, x, name='inception_v1')
    return model

def create_model_resnet18_jerett(in_shape,nb_classes):
    return classifiers.resnet_jetett.ResNet18(classes=nb_classes, input_shape=in_shape)

def create_model_resnet18_act(in_shape,nb_classes):
    return classifiers.resnet_jetett.ResNet18_act(classes=nb_classes, input_shape=in_shape)


create_model = {
    'alexnet' : create_model_alexnet,
    'alexnet_act' : create_model_alexnet_act,
    'alexnet_random' : create_model_alexnet_random,
    'raptormai': create_RaptorMai,
    'raptormai_act1' : create_RaptorMai_act1,
    'raptormai_act2' : create_RaptorMai_act2,
    'raptormai_zero' : create_RaptorMai_zero,
    'raptormai_random' : create_RaptorMai_random,
    'resnet18' : create_model_resnet18_jerett,
    'resnet18_act' : create_model_resnet18_act,
    'inceptionv1ForCifar10': create_model_inceptionv1ForCifar10_jerett,
    'inceptionv1ForCifar_act' : create_model_inceptionv1ForCifar10_act,
}
