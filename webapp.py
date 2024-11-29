from flask import Flask, render_template, request, jsonify
import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import speech_recognition as sr

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

app = Flask(__name__)

DEFAULT_MODEL_SIZE = "small"
DEFAULT_CHUNK_LENGTH = 10

ai_assistant = AIVoiceAssistant()

# Initialize Whisper model once to reuse
model_size = DEFAULT_MODEL_SIZE + ".en"
model = WhisperModel(model_size, device="cpu", compute_type="int8", num_workers=10)

@app.route('/')
def index():
    # Initial welcome message from the bot
    output = "Welcome to Botmer International. I am Emily, how can I assist you today?"
    vs.play_text_to_speech(output)
    return render_template('index.html', welcome_message=output)

def is_silence(data, max_amplitude_threshold=3000):
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

@app.route('/record', methods=['POST'])
def record():
    # Record audio from user input
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    frames = []
    for _ in range(0, int(16000 / 1024 * DEFAULT_CHUNK_LENGTH)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return jsonify({'response': "Silence detected, please speak again."})
        else:
            # Transcribe audio
            transcription = transcribe_audio(model, temp_file_path)
            #transcription = transcribe_audio_sr(temp_file_path)

            os.remove(temp_file_path)
            
            # Get response from AI assistant
            output = ai_assistant.interact_with_llm(transcription)
            if output:
                vs.play_text_to_speech(output)
                return jsonify({'response': output})
            else:
                return jsonify({'response': "I couldn't understand that, please try again."})

    except Exception as e:
        return jsonify({'response': f"Error processing audio: {e}"})

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def transcribe_audio_sr(file_path):
    # Initialize recognizer class (for recognizing the speech)
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)

    # Try recognizing the speech
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from the service; {e}"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
