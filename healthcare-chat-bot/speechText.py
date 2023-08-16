import os,sys
import speech_recognition as sr

from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

recognizer = sr.Recognizer()
f = open('/dev/null', 'w')
# sys.stderr = f

def speechToText(botname):
    # Capture audio from the microphone
    with sr.Microphone() as source:
        print(f"{botname}: Listening.....")
        audio = recognizer.listen(source)

    try:
        print(f"{botname}: Processing.....")
        text = recognizer.recognize_google(audio)
        # print("You said: " + text)

        return text
    except sr.UnknownValueError:
        print("Could not understand audio, please try ta")
    except sr.RequestError as e:
        print("Error requesting results; {0}".format(e))
    return None


def textToSpeech(text):
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")

    # Load the generated audio file
    audio = AudioSegment.from_file("output.mp3", format="mp3")

    # Play the audio
    play(audio)



