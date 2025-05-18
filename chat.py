import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load trained data
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

# Initialize model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Load intents
with open("intents.json", "r", encoding="utf-8") as json_data:
    intents = json.load(json_data)

# Chat loop
print("Let's chat! Type 'quit' to exit")
while True:
    sentence = input("You:")
    if sentence.lower() == "quit":
        print("Alex: Take care! I'm always here when you need to talk.")
        break

    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    print(f"(Debug) Input tokens: {sentence_tokens}")
    print(f"(Debug) Bag of words size: {len(X[0])}, Vector sum: {X[0].sum().item()}")

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    print(f"(Debug) Predicted tag: {tag}, Confidence: {prob.item():.4f}")  # 👈 You can remove this later

    if prob.item() > 0.5:  # Adjust as needed for stricter/flexible matching
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response = random.choice(intent["responses"])
                print(f"Alex: {response}")
    else:
        print("Alex: I'm not sure I understand. Could you say that another way?")
