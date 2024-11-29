import streamlit as st
import os
import wave
import threading
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
from rag.AIVoiceAssistant import AIVoiceAssistant
import voice_service as vs

DEFAULT_MODEL_SIZE = "small"
DEFAULT_CHUNK_LENGTH = 10

# Global flag
listening = False

# Silence detection function
def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

# Audio recording function
def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check for silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return None
        return temp_file_path
    except Exception as e:
        st.error(f"Error while processing audio: {e}")
        return None

# Transcription function
def transcribe_audio(model, file_path):
    segments, info = model.transcribe(file_path, beam_size=7)
    return ' '.join(segment.text for segment in segments)

# Audio processing loop
def listen_and_process(audio, stream, whisper_model, ai_assistant):
    global listening
    while listening:
        st.session_state.status = "Listening"
        chunk_file = record_audio_chunk(audio, stream)
        if chunk_file:
            transcription = transcribe_audio(whisper_model, chunk_file)
            os.remove(chunk_file)

            if transcription:
                st.session_state.status = "Speaking"
                response = ai_assistant.interact_with_llm(transcription)
                vs.play_text_to_speech(response)
                st.session_state.status = "Listening"

# Streamlit app
def main():
    global listening

    st.title("AI Voice Assistant")
    st.text("Welcome to the simplest voice assistant interface!")

    # Initialize session state
    if "ai_assistant" not in st.session_state:
        st.session_state.ai_assistant = AIVoiceAssistant()
    if "whisper_model" not in st.session_state:
        st.session_state.whisper_model = WhisperModel(
            DEFAULT_MODEL_SIZE + ".en", device="cpu", compute_type="int8", num_workers=4
        )
    if "audio" not in st.session_state:
        st.session_state.audio = pyaudio.PyAudio()
        st.session_state.stream = st.session_state.audio.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024
        )
    if "status" not in st.session_state:
        st.session_state.status = "Idle"

    # Display current status
    st.subheader(f"Status: {st.session_state.status}")

    # Toggle button
    if st.button("Toggle Listening"):
        if not listening:
            listening = True
            st.session_state.status = "Listening"
            threading.Thread(
                target=listen_and_process,
                args=(
                    st.session_state.audio,
                    st.session_state.stream,
                    st.session_state.whisper_model,
                    st.session_state.ai_assistant,
                ),
                daemon=True,
            ).start()
        else:
            listening = False
            st.session_state.status = "Idle"
            st.session_state.stream.stop_stream()
            st.session_state.stream.close()
            st.session_state.audio.terminate()

if __name__ == "__main__":
    main()
