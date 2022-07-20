# have to run this locally as streamlit run app.py
import streamlit as st
from Autocorrect.autocorrectreal import edit
from TestTranslation.translation import decode_sequence
from TestTranslationChinese.translation_model import decode_sequence_chinese
from AudioToText.condensedmodel import AudioToTextUsingAPI
from AudioToText.condensedmodel import AudioToTextUsingModel


st.title("FIRE COML Summer 2022 Translation Model")

option = st.selectbox("Select input type:", ("Text input", "Audio input"))
option2 = st.selectbox("Select translation language:", ("Spanish", "Chinese"))
if option == "Text input":
    input_sentence = st.text_input("Enter input sentence:")
    if input_sentence is not None and len(input_sentence) > 0:
        edited = edit(input_sentence)
        st.write("Autocorrected sentence: " + edited)
        if option2 == "Spanish":
            translated = decode_sequence(edited)[8:-5]
            st.write(translated)
            input_sentence = None
        else:
            translated = decode_sequence_chinese(edited)[8:]
            st.write(translated)
            input_sentence = None
else:
    wav_sentence = st.file_uploader("Upload an audio file (.wav):", type=\
        ["wav"])
    option3 = st.selectbox("Select audio to text model to use:", ("Our pretrained model", "Google API"))
    if st.button("Submit audio file"):
        if option3 == "Our pretrained model":
            input_list = AudioToTextUsingModel(wav_sentence)
            input_sentence = "".join(input_list)
        else:
            input_sentence = AudioToTextUsingAPI(wav_sentence)
        st.write("Raw audio to text: " + input_sentence)
        edited = edit(input_sentence)
        st.write("Autocorrected sentence: " + edited)
        if option2 == "Spanish":
            translated = decode_sequence(edited)[8:-5]
            st.write(translated)
            input_sentence = None
        else:
            translated = decode_sequence_chinese(edited)[8:]
            st.write(translated)
            input_sentence = None

