import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l_1 = nn.Linear(input_size, hidden_size)
        self.l_2 = nn.Linear(hidden_size, hidden_size)
        self.l_3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        output = self.l_1(x)
        output = self.relu(output)
        output = self.l_2(output)
        output = self.relu(output)
        output = self.l_3(output)
        return output