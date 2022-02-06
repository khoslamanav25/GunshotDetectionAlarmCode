from anyio import current_time
import sounddevice as sd
from scipy.io.wavfile import write
from data_prep import create_spectrogram
from datetime import datetime

frequency = 44100 #sample frequency of audio
duration = 4 #seconds

counter = 0

def create_spectogram_from_mic(frequency, duration, counter):
    recording = sd.rec(int(duration * frequency), 
                   samplerate=frequency, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    counter += 1
    now = datetime.now()

    current_time = now.strftime("%H_%M_%S")
    current_date = now.strftime("%d_%m_%Y")
  
    wav_file = f"mic_recorded_sounds/wav_files/wav_recording_{counter}.wav"
    spectrogram_file = f"mic_recorded_sounds/spectrograms/spectrogramsFormat/spect_{counter}_timeis_{current_time}_dateis_{current_date}"
    
    write(wav_file, frequency, recording)
    
    create_spectrogram(wav_file, spectrogram_file, train=True)
    
