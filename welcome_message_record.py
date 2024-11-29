from gtts import gTTS

# Generate the welcome message audio
welcome_message_text = "Welcome to Botmer International. I am Emily, how can I assist you today?"
tts = gTTS(welcome_message_text)
tts.save("static/audio/welcome_message.mp3")