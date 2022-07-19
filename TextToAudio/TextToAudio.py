# IMPORTANT
# !pip install gTTS

# IMPORTS
from gtts import gTTS
from IPython.display import Audio


# MAIN FUNCTION
def TextToAudio(txt_str):
  tts = gTTS(txt_str)
  audio_file = 'txtToAudio.wav'
  tts.save(audio_file)
  return audio_file