from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

genai.configure(api_key = os.getenv("GEMINI_API_KEY"))

##fucntion to load gemini pro vision
model = genai.GenerativeModel('gemini-pro-vision')
def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text


###initialize our streamlit app

st.set_page_config(page_title= "MultiLanguage Invoice Extractor")
st.header("MultiLanguage Invoice Extractor")
input = st.text_input("input prompt: ", key = "input")
uploaded_file = st.file_uploader("Choose an image of the invoice", type= ["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption = "Uploaded Image.", use_column_width = True)

submit = st.button("Tell me about the invoice")

input_prompt = """
youre an expert in extracting information from invoices.
"""
 