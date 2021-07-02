import json  #to work with JSON data
import torch
import random #to select random data(here used to select random response)
from model import NeuralNetwork
from nltk_refs import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('bot_intents.json', 'r') as json_data: #open json file..
    intents = json.load(json_data)


data = torch.load("data_file.pth")

buyItems=[] #store selected items(goods)
shelves=[] #store shelf numbers

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


document =open("Goods list.txt","w+") #finally make the selected goods list as text file
exit=0 #to exit from chatbot..
print("I'm the Chat Bot of this Super Market & I'm here to help You\n")
while True:
    if exit==1:
        break
    sentence = input("You: ")
    tempory=sentence


    sentence = tokenize(sentence) #tokenize the sentence
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]


    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                respnce=random.choice(intent['responses']) #select a random response
                print(f"{'Bot'}: {respnce}")
                if ((tag=="normal food") or (tag=="papers") or (tag=="textile") or (tag=="dairy") or (tag=="beverages") or (tag=="fruits") or (tag=="vegetables") or (tag=="medicine") or (tag=="cosmetics") or (tag=="electronic") or (tag=="plastics") or (tag=="home") or (tag=="pets") or (tag=="cleaners") or (tag=="condiments") or (tag=="baking") or (tag=="meat and fish")):
                    buyItems.append(tempory) #store the selected items
                    shelves.append(respnce) #store the selected items' shelf numbers
                elif(tag=="goodbye"):
                    exit=1
                    document.write("Your Goods list and there corresponding shelf numbers : \n\n")
                    for i, j in zip(buyItems, shelves):
                        document.write(i + " : " + j + "\n")




    else:  #if user type not related or not meaningful thing....
        print(f"{'Bot'}: Sorry!! I didn't understand!!")