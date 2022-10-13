import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from extra_keras_datasets import svhn, usps
import tensorflow as tf
import urllib3
http = urllib3.PoolManager(num_pools=50)
import cv2

class MNISTC:
    
    def __init__(self):
        self.class_num = 10
        ds_train = tfds.load(name="mnist_corrupted", split=tfds.Split.TRAIN, batch_size=-1 )
        ds_test = tfds.load(name="mnist_corrupted", split=tfds.Split.TEST, batch_size=-1)
        
        ds_train = tfds.as_numpy(ds_train) 
        ds_test = tfds.as_numpy(ds_test)
        
        self.x_train, self.y_train = ds_train["image"], ds_train["label"] # seperate the x and y
        self.x_test, self.y_test = ds_test["image"], ds_test["label"]
        
        self.x_train = self.x_train[:55000]
        self.x_val = self.x_train[55000:]
        self.y_train = self.x_train[:55000]
        self.y_val = self.x_train[55000:]
    
    def convert_data_28(self, image, label):
        image =  image.astype('float32') / 255.
        image = image.reshape((-1, 28, 28, 1))
        
        return image, label
        
    def convert_data_32(self, image, label):
        image =  image.astype('float32') / 255.
        image = image.reshape((-1, 28, 28, 1))
        image = tf.image.resize(image, (32,32))
        
        return image, label

    def test_data_28(self):
        self.x_test, self.y_test = self.convert_data_28(self.x_test, self.y_test)

        return self.x_test, self.y_test 
    
    def test_data_32(self):
        self.x_test, self.y_test = self.convert_data_32(self.x_test, self.y_test)

        return self.x_test, self.y_test 


class USPS:
    
    def __init__(self):
        self.class_num = 10
        (x_train, y_train), (x_test, y_test) = usps.load_data() # 16*16*1
        
        self.x_test = np.concatenate((x_test, x_train), axis=0)
        self.y_test = np.concatenate((y_test, y_train), axis=None)

        print(self.x_test.shape)
        print(self.y_test.shape)

    def convert_data_28(self, image, label):
        image = (np.array(image, "float32"))+1
        image = image.reshape((-1, 16, 16, 1))
        image = tf.image.resize(image, (28,28), method=	'nearest')
        
        return image, label
        
    def convert_data_32(self, image, label):
        image = (np.array(image, "float32"))+1
        image = image.reshape((-1, 16, 16, 1))
        image = tf.image.resize(image, (32,32), method=	'nearest')
        
        return image, label

    def test_data_28(self):
        self.x_test, self.y_test = self.convert_data_28(self.x_test, self.y_test)

        return self.x_test, self.y_test 
    
    def test_data_32(self):
        self.x_test, self.y_test = self.convert_data_32(self.x_test, self.y_test)

        return self.x_test, self.y_test 

class KerasMnist:
    def __init__(self):
        self.CLASS_NUM = 10

        (x_train, y_train), (self.x_test, self.y_test) = mnist.load_data()
        self.x_train = x_train[:50000]
        self.x_val = x_train[50000:]
        self.y_train = y_train[:50000]
        self.y_val = y_train[50000:]
    
    def convert_test(self, image, label):
        image =  image.astype('float32') / 255.
        image = image.reshape((-1, 28, 28, 1))

        return image, label

    def convert_train(self, image, label):
            image =  image.astype('float32') / 255.
            image = image.reshape((-1, 28, 28, 1))
            label = to_categorical(label, self.CLASS_NUM)
            
            return image, label

    def convert_test_32(self, image, label):
            image =  image.astype('float32') / 255.
            image = image.reshape((-1, 28, 28, 1))
            image = tf.image.resize(image, (32,32))
            
            return image, label

    def convert_train_32(self, image, label):
            image =  image.astype('float32') / 255.
            image = image.reshape((-1, 28, 28, 1))
            image = tf.image.resize(image, (32,32))
            label = to_categorical(label, self.CLASS_NUM)
            
            return image, label

    def train_data_aug(self):
        self.x_train, self.y_train = self.convert_train(self.x_train, self.y_train)
        self.x_val, self.y_val = self.convert_train(self.x_val, self.y_val)
        self.x_test, self.y_test = self.convert_train(self.x_test, self.y_test)
        
        datagen = ImageDataGenerator(
			featurewise_center=False,
			samplewise_center=False,
			featurewise_std_normalization=False,
			samplewise_std_normalization=False,
			zca_whitening=False,
			zca_epsilon=1e-06,
			rotation_range=0,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.,
			zoom_range=0.,
			channel_shift_range=0.,
			fill_mode='nearest',
			cval=0.,
			horizontal_flip=True,
			vertical_flip=False,
			rescale=None,
			preprocessing_function=None,
			data_format=None,
			validation_split=0.0)
        
        datagen.fit(self.x_train)
        
        return datagen, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

    def test_data_28(self):
        self.x_test, self.y_test = self.convert_test(self.x_test, self.y_test)

        return self.x_test, self.y_test 
    
    def train_data_28(self):
        self.x_train, self.y_train = self.convert_train(self.x_train, self.y_train)
        self.x_val, self.y_val = self.convert_train(self.x_val, self.y_val)
        self.x_test, self.y_test = self.convert_train(self.x_test, self.y_test)

        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test
    
    def train_data_aug_32(self):
        self.x_train, self.y_train = self.convert_train_32(self.x_train, self.y_train)
        self.x_val, self.y_val = self.convert_train_32(self.x_val, self.y_val)
        self.x_test, self.y_test = self.convert_train_32(self.x_test, self.y_test)
        
        datagen = ImageDataGenerator(
			featurewise_center=False,
			samplewise_center=False,
			featurewise_std_normalization=False,
			samplewise_std_normalization=False,
			zca_whitening=False,
			zca_epsilon=1e-06,
			rotation_range=0,
			width_shift_range=0.1,
			height_shift_range=0.1,
			shear_range=0.,
			zoom_range=0.,
			channel_shift_range=0.,
			fill_mode='nearest',
			cval=0.,
			horizontal_flip=True,
			vertical_flip=False,
			rescale=None,
			preprocessing_function=None,
			data_format=None,
			validation_split=0.0)
        
        datagen.fit(self.x_train)
        
        return datagen, self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

    def test_data_32(self):
        self.x_train, self.y_train = self.convert_test_32(self.x_train, self.y_train)
        self.x_val, self.y_val = self.convert_test_32(self.x_val, self.y_val)
        self.x_test, self.y_test = self.convert_test_32(self.x_test, self.y_test)

        return self.x_test, self.y_test # self.x_train, self.y_train, self.x_val, self.y_val, 
    
    def train_data_32(self):
        self.x_train, self.y_train = self.convert_train_32(self.x_train, self.y_train)
        self.x_val, self.y_val = self.convert_train_32(self.x_val, self.y_val)
        self.x_test, self.y_test = self.convert_train_32(self.x_test, self.y_test)

        return self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test

if __name__ == "__main__":
    m = MNISTC()
    x_train, y_train, x_val, y_val, x_test, y_test = m.train_data_28()
    print("jubin")
