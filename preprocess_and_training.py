# Load and pre-process the data
import pandas as pd
import json
import nltk
nltk.download('punkt')
import sklearn
import numpy as np
import random
import torch
import torch.nn as nn
import pickle
import math
import os

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from Convert_csv_to_JSONl import calculate_max_sentence_length
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_json('training_data.jsonl', lines=True)
#Suffling dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
# Tokenize 'text' column and store result as a separate variable
input_tokenize = data['input'].apply(lambda x:
                            word_tokenize(str(x).replace('\n', ' ').replace('\r', '').replace('\u2026', '...')))
output_tokenize = data['output'].apply(lambda x:
                            word_tokenize(str(x).replace('\n', ' ').replace('\r', '').replace('\u2026', '...')))
tokenized_text = input_tokenize + output_tokenize

# Add this to new dataframe column
data['tokenized_text'] = tokenized_text
# print(data['tokenized_text'])

# Join the tokens back into a single string for each... document?
joined_text = tokenized_text.apply(' '.join)

# Create a TfidVectorizer Object and fit_transform your joined text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(joined_text)
# print(f"Matrix shape: {tfidf_matrix.shape}")

# Splitting the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

#Further split the training data into testing and validation sets
train_data, validation_data = train_test_split(train_data, test_size=0.25, random_state=42)

#Print the sizes of the training and testing sets
# print("Training set size:", len(train_data))
# print("Testing set size:", len(test_data))
# print("Validation set size: ",len(validation_data))

# Create vocabulary mapping
vocab = {word: i for i, word in enumerate(set(word for seq in train_data['tokenized_text'] for word in seq))}
#Add <UNK> to vocab and assign it an index
vocab['<UNK>'] = len(vocab)

# Convert tokenized sequences to numerical sequences
train_sequences = [[vocab[word] for word in seq] for seq in train_data['tokenized_text']]
test_sequences = [[vocab.get(word, vocab['<UNK>']) for word in seq] for seq in test_data['tokenized_text']]
validation_sequences = [[vocab.get(word, vocab['<UNK>']) for word in seq] for seq in validation_data['tokenized_text']]

# Save vocab
vocab_file = os.path.join('/content/drive/MyDrive/Viktor', 'vocab.pkl')
with open(vocab_file, 'wb') as f:
    pickle.dump(vocab, f)
print('Vocab saved')

# Convert numerical sequences into tensors
train_sequences = [torch.tensor(seq) for seq in train_sequences]
test_sequences = [torch.tensor(seq) for seq in test_sequences]
validation_sequences = [torch.tensor(seq) for seq in validation_sequences]

# Pad sequences to have the same length
train_padded = pad_sequence(train_sequences, batch_first=True)
test_padded = pad_sequence(test_sequences, batch_first=True)
validation_padded = pad_sequence(validation_sequences, batch_first=True)

#Print shapes of padded sequences
#print("Shape of train_padded;", train_padded.shape)
#print("Shape of test_padded:", test_padded.shape)
#print("Shape of validation_padded:", validation_padded.shape)

# Get the max sentence length on the dataset. This should in theory save resources.
max_length_data = calculate_max_sentence_length()

# Positional encoding because transformers have no notion of position for the embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=max_length_data):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * -(math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
        self.scale = math.sqrt(d_model) # Normalization
    def forward(self, x):
        x = x + self.pe[:, :x.size(1):]
        return self.dropout(x)

# Defining Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, nhead, nhid, nlayers,device):
        super(TransformerModel, self).__init__()
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embedding_size)
        encoder_layers = nn.TransformerEncoderLayer(embedding_size, nhead, nhid)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size
        self.decoder = nn.Linear(embedding_size, vocab_size)
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == False,float('-inf')).masked_fill(mask == True,float(0.0))
        return mask
    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != x.size(0):
            device=x.device
            mask=self._generate_square_subsequent_mask(x.shape[1]).to(self.device)
            self.src_mask=mask
#        print(f'Self.src_mask: {mask.shape}')
        x = self.embedding(x)*math.sqrt(self.embedding_size)
 #       print("After embedding:", x.shape)
        x = self.pos_encoder(x)
  #      print("After positional encoding:", x.shape)
   #     print("Input to transformer encoder:", x.shape)
    #    print("Mask:", self.src_mask.shape)
        output = self.transformer_encoder(x, self.src_mask)
    #    print("After transformer encoder:", output.shape)
        output = self.decoder(output)
    #    print("After decoder:", output.shape)
        output = output.flatten(start_dim=1)
        return output



# Instatiate the model
model = TransformerModel(len(vocab), embedding_size=25, nhid=128, nhead=5, nlayers=5,device = device)
def init_weights(module):
    if type(module) == nn.Linear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
# Apply the weight initialization function to the model
model.apply(init_weights)

# Need the following info
#print(f'train padded shape: {train_padded.shape}')


# Training the model!!
total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params}')
if __name__ == "__main__": # So the training doesn't run when I'm actually talking to him. it.
    #Loss Function
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<UNK>"], reduction="mean")
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=.99)
    #Set up mini-batches so memory isn't that much of a constraint
    accumulation_steps = 10
    batch_size = 4
    train_data_loader = DataLoader(train_padded, batch_size = 4, shuffle=True)

    torch.manual_seed(42)
    # Set number of Epochs
    epochs = 210

    #Empty loss lists to track values
    train_loss_values = []
    test_loss_values = []
    epoch_count = []
    val_loss_values = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_i, train_batch in enumerate(train_data_loader):
            train_batch = train_batch.to(device)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            # Forward pass, compute predictions and loss
            output_train = model(train_batch)
            output_train = output_train.view(output_train.shape[0], -1)
            train_labels_shifted_left = torch.cat([train_batch[:, 1:], torch.zeros((train_batch.shape[0],1), dtype=torch.long).to(device)], dim=-1)
            #print(f'train labels shifted left shape: {train_labels_shifted_left.shape}')
            loss_val = loss_fn(output_train.view(-1, len(vocab)), train_labels_shifted_left.view(-1))
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f'{name} contains NaN values')
            # Accumulate loss
            total_loss += loss_val
            # Backward pass for accumulation steps
            if (batch_i + 1) % accumulation_steps == 0:
                total_loss = total_loss / accumulation_steps
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step() # Update weights
                optimizer.zero_grad() # Reset gradients for next batch
                total_loss = 0 # Reset total loss for next accumulation step
        del train_batch, output_train, train_labels_shifted_left, loss_val # Delete tensors to free up memory
        scheduler.step()
# for name, param in model.named_parameters():
                    #if torch.isnan(param).any():
                    #    print(f'{name} contains NaN values')
                # Clip gradients
        if epoch %5 ==0:
            print(f'Epoch {epoch}, Loss {loss_val.item()}')
            model_save_name = f'Viktor_epoch_{epoch}.pth'
            path = f"/content/drive/MyDrive/Viktor/models/{model_save_name}"
            torch.save(model.state_dict(), path)
            #Evaluation on validation set
            with torch.no_grad():
                model.to(device)
                validation_padded = validation_padded.to(device)
                model.eval()
                val_output = model(validation_padded)
                val_labels_shifted_left = torch.cat((validation_padded[:,1:], torch.zeros((validation_padded.shape[0], 1), dtype=torch.long)), dim=-1)
                val_loss = loss_fn(val_output.view(-1, len(vocab)), val_labels_shifted_left.view(-1))
                print(f'Validation Loss in epoch {epoch}: {val_loss.item()}')
                val_loss_values.append(val_loss.item())
    # Testing After training
    model.eval()
    with torch.no_grad():
        test_padded = test_padded.to(device)
        test_output = model(test_padded)
        test_labels_shifted_left = torch.cat((test_padded[:,1:], torch.zeros((test_padded.shape[0],1), dtype=torch.long)), dim=-1)
        test_loss = loss_fn(test_output.view(-1, len(vocab)), test_labels_shifted_left.view(-1))
        print(f'Test loss: {test_loss.item()}')