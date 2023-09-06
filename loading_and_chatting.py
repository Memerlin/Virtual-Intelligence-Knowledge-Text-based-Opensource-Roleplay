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
model = TransformerModel(len(vocab), embedding_size=50, nhid=2, nhead=5, nlayers=5,device = device) # Make sure these fit the args at preprocess_and_training.py
VIKTOR = 'Viktor_epoch_150.pth' #Set the model name or the path
#Load the saved parameters
model.load_state_dict(torch.load(VIKTOR, map_location=device))
#No idea why I need this below or why it needs to be inverse, but the predicted_words need it
inverse_vocab = {i: word for word, i in vocab.items()}
#Maximum length to avoid infinite loops
max_length=100
max_repeat=10
repetition_penalty = 2 # Testing how it handles this. Sept. 6: Turns out the value has to be GREATER than one. Maybe this is why he was looping so badly.
beam_width = 5 #Initialize beam width, basically how many "predictions" it does at the same time. Only the best will be kept
temperature = 1.5
while True:
    input_text = input("Human: ")
    if input_text.lower() == 'quit':
        break
    tokens = word_tokenize(input_text)
    numericalized = [vocab[token] if token in vocab else vocab["<UNK>"] for token in tokens]
    #Convert numericalized input to tensor and add batch dimension
    input_tensor = torch.tensor(numericalized, dtype=torch.int64).unsqueeze(0).to(device)
    candidates = [(input_tensor, 0)]
    predicted_words = []

    #Get input and generate
    for _ in range(max_length):
        next_candidates = []
        for candidate, score in candidates:
            output = model(candidate) 
            scaled_output = output/temperature #Apply temperature scaling to output logits
            probabilities = F.softmax(scaled_output[0], dim=-1) #Convert scaled logits to probabilities
            probabilities = probabilities[:len(vocab)] # IIRC its so it doesn't go outside the vocabulary size
            # Apply Repetition Penalty to the probabilities (from position 2 to avoid the currently predicted word and start token)
            for word_idx in set(candidate.view(-1).tolist()[2:]):
                probabilities[word_idx] *= repetition_penalty
                #Chose top probabilities
                top_probs, top_idxs = torch.topk(probabilities, beam_width)
                for i in range(beam_width):
                    next_candidate = torch.cat([candidate, top_idxs[i].unsqueeze(0).unsqueeze(0)], dim=1)
                    next_score = score - torch.log(top_probs[i]) # Convert to log-probability to avoid underflow
                    next_candidates.append((next_candidate, next_score))
                # Keep the top-k candidate sequences
                next_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = next_candidates[:beam_width] # Sort by score, not length
                candidates = next_candidates[:beam_width]
            #Select best candidates
            best_candidate = candidates[0][0]
            # Decode best seen sequence to words
            predicted_words = [inverse_vocab[int(idx)] for idx in best_candidate[0]]
        if predicted_words == "<eos>":
            break
    print('Viktor:' + ' '.join(predicted_words))