from lib2to3.pytree import convert
from msilib.schema import Directory
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import matplotlib.pyplot as plt
from path import Path
import numpy as np
import os
from glob import glob

def create_spectrogram(filename,name,train):
    plt.interactive(False) #not every plt is drawn
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    
    ax = fig.add_subplot(111)
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    
    ax.set_frame_on(False)
    
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate) #computes a spectrogram
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    if train == True:
        filename  = 'train/' + name + '.jpg'
    else:
        filename = 'test/' + name + '.jpg'
        
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)

    plt.close('all')
    
    del filename,name,clip,sample_rate,fig,ax,S
    
def convert_wav_to_spectrogram(sub_directory, train):
    directory = f'soundDatabase/{sub_directory}'
    for filename, counter in zip(os.listdir(directory), range(len(os.listdir(directory)))):
        
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            create_spectrogram(f, f'{sub_directory}_{counter}', train)
            
for i in os.listdir('soundDatabase'):
    convert_wav_to_spectrogram(i, train=True)

