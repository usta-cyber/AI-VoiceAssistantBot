import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

DEFAULT_MODEL_SIZE = "small"
DEFAULT_CHUNK_LENGTH = 10
ai_assistant = AIVoiceAssistant()
query=""
output="Welcome To Botmer International,I`mm Emily, How can i assist you Today!!"
print("Outpus:{}".format(output))
while(query!="exit"):
    query=input("Customer:")

    output = ai_assistant.interact_with_llm(query)
    print("Outpus:{}".format(output))