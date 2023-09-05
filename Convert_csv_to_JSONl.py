import pandas as pd
# Load the CSV data into a DataFrame
df = pd.read_csv('testing-dataset.csv')
# Convert the DataFrame to JSONL format
df.to_json('testing-dataset.jsonl', orient='records',
           lines = True)
