# -*- coding: utf-8 -*-
"""CondensedModel.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1d8zn9Gvp8xlKS2GRer_xzLWtm3InIBE_
"""
# IMPORTANT
# pip install pydub
# pip install SpeechRecognition
# pip install gdown
# pip install ffmpeg

# -*- coding: utf-8 -*-

# IMPORTS
import gdown
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import librosa
import speech_recognition as sr
from os.path import exists
# MODEL LOSS
def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

# BUILD MODEL
def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model

# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384

# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# GET AND INSTANTIATE MODEL
model = build_model(
    input_dim = fft_length // 2 + 1,
    output_dim = char_to_num.vocabulary_size(),
    rnn_units = 512,
)

def loadWeights():
    # PATH TO CKPT
    ckpt_link = 'https://drive.google.com/file/d/1-300ZyFUvBh1VYWyUTXhrJ9hxAJAQQcy/view?usp=sharing'

    # Set Output
    output = "AudioToTextCKPT.hdf5"

    # Download
    if not exists("AudioToTextCKPT.hdf5"):
        gdown.download(url = ckpt_link, output = output, quiet = False, fuzzy = True)

    # Load CKPT to Model
    model.load_weights(output)
    
def load_wav(filename):
    wav,_ = librosa.load(filename, sr = 22050)

    audio = tf.convert_to_tensor(
        wav,
        dtype = tf.float32
        )
    
    audio = tf.reshape(
        audio,
        shape = [audio.shape[0], 1]
    )

    return audio

# A utility function to decode the output of the network
def decode_prediction(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text

def getSpectro(wav_file):
    ###########################################
    ##  Process the Audio
    ##########################################
    audio = load_wav(wav_file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)

    spectrogram = np.expand_dims(spectrogram, axis = 0)

    return spectrogram

# Load Weights
loadWeights()

# CONVERT AUDIO TO TEXT
def AudioToTextUsingModel(wav_file):
    # Get Spectrogram
    spectro = getSpectro(wav_file)

    # Get Prediction
    pred = model.predict(spectro)

    # Get Output
    output_text = decode_prediction(pred)

    # Return Output
    return output_text

def AudioToTextUsingAPI(audio_file):
    AUDIO_FILE = audio_file
    
    # use the audio file as the audio source
    
    r = sr.Recognizer()
    
    with sr.AudioFile(AUDIO_FILE) as source:
        # reads the audio file. Here we use record instead of listen
        audio = r.record(source)  
    try:
        return r.recognize_google(audio)

    except sr.UnknownValueError:
        print(
            'Google Speech Recognition could not understand audio'
            )

    except sr.RequestError as e:
        print(
            'Could not request results from Google Speech Recognition service; {0}'.format(e)
            )

