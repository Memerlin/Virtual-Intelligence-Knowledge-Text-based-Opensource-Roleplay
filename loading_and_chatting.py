# Actual testing
import nltk
import pickle
import torch
import torch.nn.functional as F
from nltk.tokenize import word_tokenize
from preprocess_and_training import TransformerModel

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
device = torch.device('cpu')

print(f'Vocab size: {len(vocab)}')

# Load Model Architecture
model = TransformerModel(len(vocab), embedding_size=50, nhid=2, nhead=2, nlayers=5,device = device)
VIKTOR = 'Viktor_epoch_120.pth'
#Load the saved parameters
model.load_state_dict(torch.load(VIKTOR, map_location=device))
#No idea why I need this below or why it needs to be inverse, but the predicted_words need it
inverse_vocab = {i: word for word, i in vocab.items()}
#Maximum length to avoid infinite loops
max_length=100
max_repeat=2 # Maximum allowed repeat length

while True:
    input_text = input("Human: ")
    if input_text.lower() == 'quit':
        break
    tokens = word_tokenize(input_text)
    numericalized = [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]
    #Convert numericalized input to tensor and add batch dimension
    input_tensor = torch.tensor(numericalized, dtype=torch.int64).unsqueeze(0).to(device)
    predicted_words = []
    temperature = .4
    #Get input and generate
    for _ in range(max_length):
        #Forward pass through the model
        output = model(input_tensor)
        #Apply temperature scaling to output logits
        scaled_output = output/temperature
        #Convert scaled logits to probabilities
        probabilities = F.softmax(scaled_output[0], dim=-1)
        probabilities = probabilities[:len(vocab)]
        #Sample from these probabilities to get our predicted text
        predicted_index = torch.multinomial(probabilities, num_samples=1).item()
        #convert index to word and add it to our list of predicted words
        predicted_word = inverse_vocab[predicted_index]
        predicted_words.append(predicted_word)
        #Check if the predicted word is a punctuation mark.
        if predicted_words[-max_repeat:].count(predicted_word) == max_repeat:
            break
        #Add the predicted index to our input tensor for the next round
        input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_index]],
                                                            dtype=torch.int64)], dim=1)
    print('Viktor:' + ' '.join(predicted_words))