from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import cv2
import data
import model as M
import argparse

BATCH_SIZE = 32

data_input_shape = {
    'mnist'     : (28,28,1),
    'mnist32'   : (32,32,1),
}

def train(model, x_train, y_train, x_val, y_val, x_test, y_test, model_name, type):
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	model_checkpoint = ModelCheckpoint(
		filepath='%s_ep{epoch:02d}_loss{loss:.4f}_val_loss{val_loss:.4f}.h5' % ('./save/'),
		monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

	terminate_on_nan = TerminateOnNaN()
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1) # , patience=5, verbose=1

	model.summary()

	callbacks = [model_checkpoint,
			learning_rate_scheduler,
			terminate_on_nan, early_stopping]

	print(x_train.shape)
	print(y_train.shape)

	# trainig
	model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=100,
                  validation_data=(x_val, y_val),
                  shuffle=True,
                  callbacks=callbacks)

	score = model.evaluate(x_test, y_test, verbose=1)
	model.save("model/"+model_name+"_2_mnist.h5")

	print(score[0], score[1])

def train_aug(model, datagen, x_train, y_train, x_val, y_val, x_test, y_test, model_name, type):
	
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	model_checkpoint = ModelCheckpoint(
		filepath='%s_ep{epoch:02d}_loss{loss:.4f}_val_loss{val_loss:.4f}.h5' % ('./save/'),
		monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

	learning_rate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

	terminate_on_nan = TerminateOnNaN()
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1) # , patience=5, verbose=1

	model.summary()

	callbacks = [model_checkpoint,
			learning_rate_scheduler,
			terminate_on_nan, early_stopping]

	print(x_train.shape)
	print(y_train.shape)
	# trainig
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            validation_data=(x_val, y_val),
                            epochs=100, verbose=1,
                            callbacks=callbacks)

	score = model.evaluate(x_test, y_test, verbose=1)
	model.save("model/"+model_name+"_mnist.h5")

	print(score[0], score[1])


if __name__ == "__main__":
    model = ["alexnet"] # "resnet18", "lenet1", "lenet5"
    

    for model_name in model:

        d = data.KerasMnist()

        input_32 = {
            'lenet5' : False,
            'lenet5_aug' : False,
            'lenet5_act' : False,
            'lenet5_random' : False,
            'lenet1' : False,
            'lenet1_act' : False,
            'lenet1_random' : False,
            'lenet1_aug' : False,
            'resnet18' : False,
            'resnet18_aug' : False,
            'resnet18_act' : False,
            'resnet18_random' : False,
            'resnet20' : True,
            'simplenet' : True,
            'simplenet_aug' : True,
            'simplenet_act' : True,
            'simplenet_random' : True,
            'alexnet' : True,
            'alexnet_aug' : True,
            'alexnet_act' : True,
            'alexnet_random' : True 
        }

        input_aug = {
            'lenet5' : False,
            'lenet5_aug' : True,
            'lenet5_act' : False,
            'lenet5_random' : False,
            'lenet1' : False,
            'lenet1_aug' : True,
            'lenet1_act' : False,
            'lenet1_random' : False,
            'resnet18' : False,
            'resnet18_aug' : True,
            'resnet18_act' : False,
            'resnet18_random' : False,
            'resnet20' : False,
            'simplenet' : False,
            'simplenet_aug' : True,
            'simplenet_act' : False,
            'simplenet_random' : False,
            'alexnet' : False,
            'alexnet_aug' : True,
            'alexnet_act' : False,
            'alexnet_random' : False 
        }

        if input_32[model_name] == False:
            model = M.create_model[model_name](data_input_shape['mnist'],10)
            if input_aug[model_name] == False:
                x_train, y_train, x_val, y_val, x_test, y_test = d.train_data_28()
                train(model, x_train, y_train, x_val, y_val, x_test, y_test , model_name, type)
            else:
                datagen, x_train, y_train, x_val, y_val, x_test, y_test = d.train_data_aug()
                train_aug(model, datagen, x_train, y_train, x_val, y_val, x_test, y_test , model_name, type)
        else :
            model = M.create_model[model_name](data_input_shape['mnist32'],10)
            if input_aug[model_name] == False:
                x_train, y_train, x_val, y_val, x_test, y_test = d.train_data_32()
                train(model, x_train, y_train, x_val, y_val, x_test, y_test , model_name, type)
            else:
                datagen, x_train, y_train, x_val, y_val, x_test, y_test = d.train_data_aug_32()
                train_aug(model, datagen, x_train, y_train, x_val, y_val, x_test, y_test , model_name, type) 


        model1 = tf.keras.models.load_model("model/"+model_name+"_mnist.h5")

        features1 = model1.predict(x_test)

        y_test = y_test.argmax(axis=1)
        y_pred1 = features1.argmax(axis=1)

    