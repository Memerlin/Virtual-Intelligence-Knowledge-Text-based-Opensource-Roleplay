import pandas as pd
from nltk.tokenize import word_tokenize
if __name__ == "__main__":
    # Load the CSV data into a DataFrame
    df = pd.read_csv('testing-dataset.csv')
    # Convert the DataFrame to JSONL format
    df.to_json('testing-dataset.jsonl', orient='records',
            lines = True)
# Calculating the length of each sentence
def calculate_max_sentence_length():
    df = pd.read_json('testing-dataset.jsonl', lines=True)
    df['sentence_length'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    max_sentence_length = df['sentence_length'].max()
    return max_sentence_length
print("Maximum sentence length is: ", calculate_max_sentence_length())