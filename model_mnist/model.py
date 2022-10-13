from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import glorot_normal, RandomNormal
from tensorflow.keras import regularizers
import tensorflow as tf
import classifiers.resnet_jetett

def create_model_resnet18(in_shape,nb_classes):
    return classifiers.resnet_jetett.ResNet18(classes=nb_classes, input_shape=in_shape)

def create_model_resnet18_act(in_shape,nb_classes):
    return classifiers.resnet_jetett.ResNet18_act(classes=nb_classes, input_shape=in_shape)

def create_model_resnet18_random(in_shape,nb_classes):
    return classifiers.resnet_jetett.ResNet18_random(classes=nb_classes, input_shape=in_shape)

def create_model_Lenet5(in_shape, nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 2 :    # Conv Layer 2 + # Pooling Layer 2 + Flatten
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    # Layer 3 :   # Fully connected layer 1
    model.add(Dense(units=120, activation='relu'))
    # Layer 4:    # Fully connected layer 2
    model.add(Dense(units=84, activation='relu'))
    # Layer 5:    # Output Layer
    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

def create_model_Lenet5_act(in_shape, nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='tanh', input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 2 :    # Conv Layer 2 + # Pooling Layer 2 + Flatten
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='tanh', input_shape=(14, 14, 6)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    # Layer 3 :   # Fully connected layer 1
    model.add(Dense(units=120, activation='relu'))
    # Layer 4:    # Fully connected layer 2
    model.add(Dense(units=84, activation='relu'))
    # Layer 5:    # Output Layer
    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

def create_model_Lenet5_random(in_shape, nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(filters=6, kernel_size=5, strides=1, activation='relu', input_shape=in_shape, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    # Layer 2 :    # Conv Layer 2 + # Pooling Layer 2 + Flatten
    model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Flatten())
    # Layer 3 :   # Fully connected layer 1
    model.add(Dense(units=120, activation='relu'))
    # Layer 4:    # Fully connected layer 2
    model.add(Dense(units=84, activation='relu'))
    # Layer 5:    # Output Layer
    model.add(Dense(units=nb_classes, activation='softmax'))
    return model

def create_model_Lenet1(in_shape,nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(20, 5, padding="same", activation='relu', input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, 5, padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # define the second FC layer
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

def create_model_Lenet1_act(in_shape,nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(20, 5, padding="same", activation='tanh', input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, 5, padding="same", activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # define the second FC layer
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

def create_model_Lenet1_random(in_shape,nb_classes):
    model = Sequential()
    # Layer 1 :  # Conv Layer 1 + # Pooling layer 1
    model.add(Conv2D(20, 5, padding="same", activation='relu', input_shape=in_shape, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(50, 5, padding="same", activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    # define the second FC layer
    model.add(Dense(nb_classes, activation='softmax'))
    
    return model

def create_model_Simplenet(in_shape, nb_classes, s = 2, weight_decay = 1e-2, act="relu"):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal(), input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 2
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 4
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    
    
    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    # Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    
    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    # Final Classifier
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def create_model_Simplenet_act(in_shape, nb_classes, s = 2, weight_decay = 1e-2, act="tanh"):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal(), input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 2
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 4
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    
    
    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=RandomNormal(stddev=0.01)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    # Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    
    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=glorot_normal()))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    # Final Classifier
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))

    return model

def create_model_Simplenet_random(in_shape, nb_classes, s = 2, weight_decay = 1e-2, act="relu"):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.), input_shape=in_shape))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 2
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 4
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # First Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))
    
    
    # Block 5
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 6
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 7
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    # Second Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    
    # Block 8
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 9
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))
    # Third Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    
    
    # Block 10
    model.add(Conv2D(512, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(Dropout(0.2))

    # Block 11  
    model.add(Conv2D(2048, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation(act))
    model.add(Dropout(0.2))
    
    # Block 12  
    model.add(Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation(act))
    # Fourth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))
    model.add(Dropout(0.2))


    # Block 13
    model.add(Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.005), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1.)))
    model.add(Activation(act))
    # Fifth Maxpooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=s))

    # Final Classifier
    model.add(Flatten())
    model.add(Dense(nb_classes, activation='softmax'))

    return model

my_activation = 'relu'

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


create_model = {
    'resnet18' : create_model_resnet18,
    'resnet18_aug' : create_model_resnet18,
    'resnet18_act' : create_model_resnet18_act,
    'resnet18_random' : create_model_resnet18_random,
    'lenet5' : create_model_Lenet5,
    'lenet5_aug' : create_model_Lenet5, 
    'lenet5_act': create_model_Lenet5_act,
    'lenet5_random' : create_model_Lenet5_random,
    'lenet1' : create_model_Lenet1,
    'lenet1_aug' : create_model_Lenet1,
    'lenet1_act' : create_model_Lenet1_act,
    'lenet1_random' : create_model_Lenet1_random,
    'simplenet' : create_model_Simplenet,
    'simplenet_aug' : create_model_Simplenet,
    'simplenet_act' : create_model_Simplenet_act,
    'simplenet_random' : create_model_Simplenet_random,
    'alexnet' : create_model_alexnet,
    'alexnet_aug' : create_model_alexnet,
    'alexnet_act' : create_model_alexnet_act,
    'alexnet_random' : create_model_alexnet_random 
}