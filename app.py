import streamlit as st
from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY was not loaded from .env")
    st.stop()

client = genai.Client(api_key=api_key)


def get_gemini_response(question):
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=question
    )
    return response.text


st.header("Gemini Application")

with st.form("gemini_form"):
    input_text = st.text_input("Enter your question:")
    submit = st.form_submit_button("Submit")

if submit:
    if input_text:
        response = get_gemini_response(input_text)
        st.write(response)
    else:
        st.warning("Please enter a question.")
        
