from statistics import mode
from keras_preprocessing.image import ImageDataGenerator
import os

batch_size = 64

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

other_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'spectrogram_sorted_train',  # this is the train directory
        target_size=(64, 64),  # all images will be resized to 64, 64
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

validation_generator = other_datagen.flow_from_directory(
        'spectrogram_sorted_test',
        target_size=(64, 64),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)


from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint

model_parameters = {
                    'batch_size' : 64,
                    'epochs' : 200,
                    'model_checkpoint' : 'best_model.hdf5',
                    'learning_rate' : 0.0005,
                    'decay' : 1e-6,
                    'train' : False
                }

def create_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64,64,3)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizers.RMSprop(lr=model_parameters['learning_rate'], decay=model_parameters['decay']), 
                        loss="categorical_crossentropy",metrics=["accuracy"])

        return model

model = create_model()
model.summary()

checkpoint = ModelCheckpoint(model_parameters['model_checkpoint'], monitor='val_accuracy', verbose=1, 
                             save_best_only=True, mode='auto', period=1, save_weights_only=False)

from keras.models import load_model

if os.path.isfile(model_parameters['model_checkpoint']):      
        model = load_model(model_parameters['model_checkpoint'])


if model_parameters['train'] == True:
        model.fit_generator(train_generator, steps_per_epoch=2000 // model_parameters['batch_size'], 
                        epochs=model_parameters['epochs'], validation_data=validation_generator, 
                        validation_steps=800 // model_parameters['batch_size'], callbacks=[checkpoint], verbose=1)

