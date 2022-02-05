import sounddevice as sd

from scipy.io.wavfile import write

import wavio as wv

frequency = 44100 #sample frequency of audio
duration = 4 #seconds

while True:
    recording = sd.rec(int(duration * frequency), 
                   samplerate=frequency, channels=2)
    # Record audio for the given number of seconds
    sd.wait()
    write("recording0.wav", frequency, recording)