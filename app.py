# have to run this locally as streamlit run app.py
import streamlit as st
from translation_test import *
st.title("Translation model test")
input_sentence = st.text_input("Enter input sentence:")
if input_sentence is not None and len(input_sentence) > 0:
    translated = decode_sequence(input_sentence)
    st.write(translated)
    input_sentence = None