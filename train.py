#this train.py file is to train the model...
import numpy as np #to working with arrays..
import json #to work with JSON data
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_refs import bag_of_words, tokenize, stem
from model import NeuralNetwork #this is a the class that in model.py...

with open('bot_intents.json', 'r') as f: #open the json file..
    intents = json.load(f)

all_the_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_the_words.extend(w)
        xy.append((w, tag))

ignoring_words = [',', '?', '.', '!', '@'] #ignoring characters...
all_the_words = [stem(w) for w in all_the_words if w not in ignoring_words]
# to remove duplicates and sorting....
all_the_words = sorted(set(all_the_words))
tags = sorted(set(tags))


train_x = [] #train data..
train_y = [] #train data..
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_the_words)
    train_x.append(bag)
    label = tags.index(tag)
    train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)

# below are the hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(train_x[0])
hidden_size = 8
output_size = len(tags)

class SetOfChatData(Dataset):

    def __init__(self):
        self.n_samples = len(train_x)
        self.x_data = train_x
        self.y_data = train_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = SetOfChatData()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# this is for train the model......
for epo in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

data = {"model_state": model.state_dict(),"input_size": input_size,"hidden_size": hidden_size,"output_size": output_size,"all_words": all_the_words, "tags": tags}
torch.save(data, "data_file.pth") #saved the file after training...
print(f'Training complete successfully. file saved to {"data_file.pth"}') #after training complete for model...
print("Now run the chat.py file to get the chat bot...")