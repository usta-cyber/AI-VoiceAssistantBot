import streamlit as st
from rag.AIVoiceAssistant import AIVoiceAssistant

# Initialize the AI assistant once and store it in session state
if "ai_assistant" not in st.session_state:
    st.session_state.ai_assistant = AIVoiceAssistant()

# Create a session state to store the conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Streamlit UI
st.title("Conversational Chatbot")
st.subheader("Welcome to Botmer International!")
st.text("I'm Emily. How can I assist you today?")

# User input
query = st.text_input("Your Message:", "")

# Process user input
if st.button("Send"):
    if query.lower() == "exit":
        st.write("Thank you for chatting with me. Goodbye!")
    elif query.strip() != "":
        # AI assistant response
        response = st.session_state.ai_assistant.interact_with_llm(query)

        # Store the interaction in the conversation history
        st.session_state.conversation_history.append(("You", query))
        st.session_state.conversation_history.append(("Emily", response))
        print("sss",response)
        # Clear the input box
        st.rerun()

# Display the conversation history
for speaker, message in st.session_state.conversation_history:
    if speaker == "You":
        st.markdown(f"**{speaker}:** {message}")
    else:
        st.markdown(f"_{speaker}:_ {message}")
