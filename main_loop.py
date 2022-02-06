from convnet_model import other_datagen, model, train_generator
from microphone_inputs import create_spectogram_from_mic
from data_prep import clear_folder
import numpy as np

directory = 'mic_recorded_sounds/spectrograms'

clear_folder('mic_recorded_sounds/spectrograms/spectrogramsFormat')
clear_folder('mic_recorded_sounds/wav_files')

def get_generator():
    new_image_generator = other_datagen.flow_from_directory(
            directory=directory,
            target_size=(64, 64),
            batch_size=64,
            shuffle=False
    )
    return new_image_generator

running = True

frequency = 44100 #sample frequency of audio
duration = 4 #seconds

counter = 0


while running == True:
    
    
    print('* * Starting to Record * *')
    
    create_spectogram_from_mic(frequency, duration, counter)
    
    new_image_generator = get_generator() #has to be redefined every time spectrogram is created
    
    new_image_generator.reset()
    pred = model.predict_generator(new_image_generator, steps=10, verbose=0)
    predicted_class_indices=np.argmax(pred,axis=1)
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    
    if 'gun_shot' in predictions[:-1]:
            print('ALARM ALARM ALARM')
        
        
    counter += 1
    
    if counter >= 10:
        counter = 0
        
        clear_folder('mic_recorded_sounds/spectrograms/spectrogramsFormat')
        clear_folder('mic_recorded_sounds/wav_files')
        print('( RESETING PREDICTIONS )')
        
    print(f'----- Prediction Is {predictions[:-1]} -----')
    print(f'All Predictions {predictions}: ')
    print('* * New Recording * *')

    
