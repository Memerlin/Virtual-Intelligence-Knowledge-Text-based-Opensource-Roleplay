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
model = TransformerModel(len(vocab), embedding_size=50, nhid=2, nhead=2, nlayers=5,device = device) # Make sure these fit the args at preprocess_and_training.py
VIKTOR = 'Viktor_epoch_150.pth' #Set the model name or the path
#Load the saved parameters
model.load_state_dict(torch.load(VIKTOR, map_location=device))
#No idea why I need this below or why it needs to be inverse, but the predicted_words need it
inverse_vocab = {i: word for word, i in vocab.items()}
#Maximum length to avoid infinite loops
max_length=100
max_repeat=10
repetition_penalty = 0.5 # Testing how it handles this

while True:
    input_text = input("Human: ")
    if input_text.lower() == 'quit':
        break
    tokens = word_tokenize(input_text)
    numericalized = [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]
    #Convert numericalized input to tensor and add batch dimension
    input_tensor = torch.tensor(numericalized, dtype=torch.int64).unsqueeze(0).to(device)
    predicted_words = []
    temperature = .1
    #Get input and generate
    for _ in range(max_length):
        #Forward pass through the model
        output = model(input_tensor)
        #Apply temperature scaling to output logits
        scaled_output = output/temperature
        #Convert scaled logits to probabilities
        probabilities = F.softmax(scaled_output[0], dim=-1)
        probabilities = probabilities[:len(vocab)] # IIRC its so it doesn't go outside the vocabulary size
        # Apply Repetition Penalty to the probabilities
        for i, word in enumerate(predicted_words[-max_repeat:]):
            word_index = vocab[word]
            probabilities[word_index] *= repetition_penalty ** (max_repeat - i)
        predicted_index = torch.multinomial(probabilities, num_samples=1).item() #Sample from these probabilities to get our predicted text
        # Convert index to word and add it to our list of predicted words
        predicted_word = inverse_vocab[predicted_index]
        predicted_words.append(predicted_word)
        #Add the predicted index to our input tensor for the next round
        input_tensor = torch.cat([input_tensor, torch.tensor([[predicted_index]],
                                                            dtype=torch.int64)], dim=1)
        # Check if predicted word is <eos> token, and stop if it is.
        if predicted_word == "<eos>":
            break
    print('Viktor:' + ' '.join(predicted_words))