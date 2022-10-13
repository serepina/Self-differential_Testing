from tokenize import group
import tensorflow as tf
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
import diff as di
import argparse

BATCH_SIZE = 32

data_input_shape = {
    'mnist'     : (28,28,1),
    'cifar10'   :(32,32,3),
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
	model.save("model/"+model_name+"_"+type+"_cifar10.h5")

	print(score[0], score[1])

def train_aug(model, datagen, x_train, y_train, x_val, y_val, x_test, y_test, model_name):
	
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

	# trainig
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                            validation_data=(x_val, y_val),
                            epochs=100, verbose=1,
                            callbacks=callbacks)

	score = model.evaluate(x_test, y_test, verbose=1)
	model.save("model/"+model_name+"_cifar_aug.h5")

	print(score[0], score[1]) 

def load_model(model_name, x_test):
    model = tf.keras.models.load_model("model/"+model_name+"_cifar.h5")
    features = model.predict(x_test)
    y_pred = features.argmax(axis=1)

    return y_pred

if __name__ == "__main__":
	
	model = ["inceptionv1ForCifar10"] # "raptormai", "alexnet", "resnet18"

	for model_name in model:
		aug_model = False
		
		model = M.create_model[model_name](data_input_shape['cifar10'],10)
		d = data.KerasCifar10()
			
		if aug_model == False:
			x_train, y_train, x_val, y_val, x_test, y_test = d.train_data()
			train(model, x_train, y_train, x_val, y_val, x_test, y_test , model_name, type)
		else :
			datagen, x_train, y_train, x_val, y_val, x_test, y_test = d.train_data_aug()
			train_aug(model, datagen, x_train, y_train, x_val, y_val, x_test, y_test , model_name)
			
		y_test = y_test.argmax(axis=1)
		y_pred1 = load_model(model_name,x_test)
		print(di.acc(y_test, y_pred1))