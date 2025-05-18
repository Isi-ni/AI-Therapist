import torch
import torch.nn as nn  

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Added Dropout Layer

        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.3)  # Added Dropout Layer

        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout1(out)  # Added Dropout

        out = self.l2(out)
        out = self.relu(out)
        out = self.dropout2(out)  # Added Dropout

        out = self.l3(out)
        return out
