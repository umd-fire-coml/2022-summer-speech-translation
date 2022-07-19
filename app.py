# have to run this locally as streamlit run app.py
import streamlit as st
from Autocorrect.autocorrectreal import edit
from TestTranslation.translation import *
from TestTranslationChinese.translation_model import decode_sequence_chinese


st.title("Translation model test")

option = st.selectbox("Select input type:", ("text input", "audio input"))
option2 = st.selectbox("Select translation language:", ("Spanish", "Chinese"))
if option == "text input":
    input_sentence = st.text_input("Enter input sentence:")
    if input_sentence is not None and len(input_sentence) > 0:
        if option2 == "Spanish":
            edited = edit(input_sentence)
            st.write("Autocorrected sentence: " + edited)
            translated = decode_sequence(edited)[8:-5]
            st.write(translated)
            input_sentence = None
        else:
            edited = edit(input_sentence)
            st.write("Autocorrected sentence: " + edited)
            translated = decode_sequence_chinese(edited)[8:]
            st.write(translated)
            input_sentence = None
else:
    wav_sentence = st.file_uploader("Upload a wav file:")
    st.button("Submit wav file")


