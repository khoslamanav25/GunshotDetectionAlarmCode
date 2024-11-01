import sounddevice as sd
import numpy as np
import requests
import pyaudio
import wave
import time
from scipy.io.wavfile import write

SAMPLERATE = 44100
DURATION = 2
HEROKU_URL = "https://gda.herokuapp.com/predict"
ALARM_FILE = "alarm.wav"
MICROPHONE_DEVICE_INDEX = 1
SPEAKER_DEVICE_INDEX = 2

def record_audio(filename="audio_sample.wav"):
    audio = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype='int16', device=MICROPHONE_DEVICE_INDEX)
    sd.wait()
    write(filename, SAMPLERATE, audio)
    return filename

def send_to_model(filename):
    with open(filename, 'rb') as f:
        response = requests.post(HEROKU_URL, files={'file': f})
    if response.status_code == 200:
        return response.json()
    return None

def play_alarm():
    wf = wave.open(ALARM_FILE, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True, output_device_index=SPEAKER_DEVICE_INDEX)
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    stream.stop_stream()
    stream.close()
    p.terminate()

while True:
    audio_file = record_audio()
    result = send_to_model(audio_file)
    if result and result.get('gunshot_detected'):
        play_alarm()
    time.sleep(DURATION)
