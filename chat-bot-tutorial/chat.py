from os import system
import random
import json
from tkinter.filedialog import Open
import torch
from model import NeuralNet
from nltik_utils import bag_of_words,tokenize
from speechText import speechToText, textToSpeech

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json",'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def predict(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence,all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)
    
    return model(X)

# print(speechToText())
"a".strip()
bot_name = "CityGuide"
print("Let's chat! type '!mic' to input via microphone \n'!repeat' to hear again \n'!quit' to exit")
response = ""
while True:
    print("You: ",end="")
    sentence = input()
    if sentence == "!quit":
        break
    elif sentence == "!repeat":
        textToSpeech(response)
        continue
    elif sentence == "!mic":
        sentence = speechToText(bot_name)
        if sentence == None:
            response = "Could not understand audio, please try again!"
        else:
            print(f"{bot_name}: You said, {sentence}")
    if sentence != None and len(sentence.strip()) > 0:
        output = predict(sentence)
        _, predicted = torch.max(output,dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output,dim=1)
        prob = probs[0][predicted.item()]

        response = ""
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    response = random.choice(intent['responses'])
        else:
            response = "I do not understand...."

    print(f"{bot_name}: {response}")
    textToSpeech(response)



