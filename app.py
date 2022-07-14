# have to run this locally as streamlit run app.py
import streamlit as st

from TestTranslation.translation import *



st.title("Translation model test")

option = st.selectbox("Select input type:", ("text input", "audio input"))
if option == "text input":
    input_sentence = st.text_input("Enter input sentence:")
    if input_sentence is not None and len(input_sentence) > 0:
        translated = decode_sequence(input_sentence)
        st.write(translated)
        input_sentence = None
else:
    wav_sentence = st.file_uploader("Upload a wav file:")
    st.button("Submit wav file")


