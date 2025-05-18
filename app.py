from flask import Flask
from flask_socketio import SocketIO, emit
import torch
import json
import random
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Load intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load model data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "AuroraBot"

@socketio.on('user_message')
def handle_user_message(data):
    sentence = data['message']
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).float().unsqueeze(0)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                emit('bot_response', {'message': response})
                return
    else:
        emit('bot_response', {'message': "I'm not sure how to respond. Can you try again?"})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
