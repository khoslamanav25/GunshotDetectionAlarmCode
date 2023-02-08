from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model



def ann_model_1(input_shape):
    
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=input_shape))
    model.add(Dense(750, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.summary()
    
    return model

