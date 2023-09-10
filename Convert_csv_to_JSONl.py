import pandas as pd

from nltk.tokenize import word_tokenize

if __name__ == "__main__":
    # load DataFrame
    df = pd.read_pickle('new_format_training_data.pkl')
    list(df.columns)
    #Making sure everything is a string
    df['bot_name'] = df['bot_name'].astype(str)
    df['bot_definitions'] = df['bot_definitions'].astype(str)
    df['chat_examples'] = df['chat_examples'].astype(str)
    df['bot_greeting'] = df['bot_greeting'].astype(str)
    df['conversation'] = df['conversation'].astype(str)

# Calculating the length of each sentence
def calculate_max_sentence_length():
    data = pd.read_pickle('new_format_training_data.pkl')
    bot_name_length = data['bot_name'].apply(lambda x: len(word_tokenize(x)))
    bot_definitions_length = data['bot_definitions'].apply(lambda x: len(word_tokenize(x)))
    chat_examples_length = data['chat_examples'].apply(lambda x: len(word_tokenize(x)))
    bot_greeting_length = data['bot_greeting'].apply(lambda x: len(word_tokenize(x)))
    conversation_length = data['conversation'].apply(lambda x: len(word_tokenize(x)))
    data['sentence_length'] = bot_name_length + bot_definitions_length + bot_greeting_length + conversation_length + chat_examples_length
    max_sentence_length = data['sentence_length'].max()
    return max_sentence_length

print("Maximum sentence length is: ", calculate_max_sentence_length())