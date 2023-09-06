import pandas as pd
from nltk.tokenize import word_tokenize
if __name__ == "__main__":
    # Load the CSV data into a DataFrame
    df = pd.read_csv('training-data2.csv', header=0)
    #Turns out I actually need an end of sentence token
    eos_token = '<eos>'
    #Making sure everything is a string
    df['input'] = df['input'].astype(str)
    df['output'] = df['output'].astype(str)
    df[['input', 'output']] = df[['input', 'output']].applymap(lambda x: x + '{} <eos>'.format(x))
    # Convert the DataFrame to JSONL format
    with open('training-data2.jsonl', 'w', encoding='utf-8') as f:
        df.to_json(f, orient='records', lines=True, force_ascii=False, date_format='iso')

# Calculating the length of each sentence
def calculate_max_sentence_length():
    data = pd.read_json('training-data2.jsonl', lines=True, encoding='utf-8')
    input_length = data['input'].apply(lambda x: len(word_tokenize(x)))
    output_length = data['output'].apply(lambda x: len(word_tokenize(x)))
    data['sentence_length'] = input_length + output_length
    max_sentence_length = data['sentence_length'].max()
    return max_sentence_length
print("Maximum sentence length is: ", calculate_max_sentence_length())