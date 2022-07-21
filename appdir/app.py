# have to run this locally as streamlit run app.py
import streamlit as st
import os
from Autocorrect.autocorrectreal import edit
from TestTranslation.translation import decode_sequence
from TestTranslationChinese.translation_model import decode_sequence_chinese
from AudioToText.condensedmodel import AudioToTextUsingAPI
from AudioToText.condensedmodel import AudioToTextUsingModel
from TextToAudio.TextToTalkingFace import *



st.title("FIRE COML Summer 2022 Translation Model")

option = st.selectbox("Select input type:", ("Text input", "Audio input"))
option2 = st.selectbox("Select translation language:", ("Spanish", "Chinese"))
option4 = st.selectbox("Select visualization face:", ('angelina', 'anne', 'audrey', 'aya', 'cesi', 'dali',
                 'donald', 'dragonmom', 'dwayne', 'harry', 'hermione',
                 'johnny', 'leo', 'morgan', 'natalie', 'neo', 'obama',
                 'rihanna', 'ron', 'scarlett', 'taylor'))
st.write("Note: video can take up to a minute to load")
if option == "Text input":
    input_sentence = st.text_input("Enter input sentence:")
    if input_sentence is not None and len(input_sentence) > 0:
        edited = edit(input_sentence)
        st.write("Autocorrected sentence: " + edited)
        if option2 == "Spanish":
            translated = decode_sequence(edited)[8:-5]
            st.write(translated)
        else:
            translated = decode_sequence_chinese(edited)[8:]
            st.write(translated)
        print(os.getcwd())
        output_path = TextToTalkingFace(translated, option4)
        st.video(output_path, format="video/mp4", start_time=0)
        deleteOldFiles(output_path)
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
        else:
            translated = decode_sequence_chinese(edited)[8:]
            st.write(translated)
        output_path = TextToTalkingFace(translated, option4)
        st.video(output_path, format="video/mp4", start_time=0)
        deleteOldFiles(output_path)
        input_sentence = None

