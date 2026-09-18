
from dotenv import load_dotenv

import streamlit as st
import os
import google.generativeai as genai

# Load .env file
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Change gemini-pro because it is no longer available
model = genai.GenerativeModel("gemini-3.6-flash")

chat = model.start_chat(history=[])


def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response


## initialize our streamlit app

st.set_page_config(page_title="Q&A ChatBot")
st.header("Gemini LLM Application")


## Initialize session state for the chat history if it does not exist

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


input = st.text_input("Input:", key="input")

submit = st.button("Ask the question")


if submit and input:

    response = get_gemini_response(input)

    ## add user query to chat history

    st.session_state["chat_history"].append(("You", input))

    st.subheader("The response is")

    bot_response = ""

    for chunk in response:
        if chunk.text:
            st.write(chunk.text)
            bot_response += chunk.text

    # Add complete bot response to chat history
    st.session_state["chat_history"].append(("Bot", bot_response))


    st.subheader("Chat history is")

    for role, text in st.session_state["chat_history"]:
        st.write(f"{role}: {text}")

