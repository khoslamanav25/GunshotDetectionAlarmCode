import librosa as lb
import numpy as np



def mfcc_feature_extractor(path):
    data, simple_rate = lb.load(path, res_type='kaiser_fast')
    data = lb.feature.mfcc(data, n_mfcc=128)
    data = np.mean(data,axis=1)
    return data


def stft_feature_extractor(path):
    data, sample_rate = lb.load(path, res_type='kaiser_fast')
    stft = np.abs(lb.stft(data, n_fft=128))
    stft_features = np.mean(stft, axis=1)
    return stft_features



