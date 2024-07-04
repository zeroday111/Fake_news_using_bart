import requests
import pandas as pd
# import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Step 1: Collect News Headlines using NewsAPI
API_KEY = 'b44e88a74a3e49648c54dffbb7cffd32'
url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'

response = requests.get(url)
data = response.json()

headlines = [article['title'] for article in data['articles']]

def save_headlines_to_file(headlines, filename='headlines.csv'):
    df = pd.DataFrame(headlines, columns=['headline'])
    df.to_csv(filename, index=False)

save_headlines_to_file(headlines)

# Step 2: Load BERT model and classify headlines
# Load saved model and tokenizer
# model_dir = './saved_model'
# model = BertForSequenceClassification.from_pretrained(model_dir)
# tokenizer = BertTokenizer.from_pretrained(model_dir)

# # Function to test the model with new text input
# def test_model(input_text, model, tokenizer, max_length, device):
#     model.eval()
#     inputs = tokenizer(input_text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
#     inputs = {key: value.to(device) for key, value in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#         prediction = torch.argmax(outputs.logits, dim=-1).item()
#     return 'Fake News' if prediction == 1 else 'Real News'

# # Load headlines
# headlines_df = pd.read_csv('headlines.csv')

# # Initialize device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# # Predict on headlines
# headlines_df['prediction'] = headlines_df['headline'].apply(lambda x: test_model(x, model, tokenizer, max_length=128, device=device))

# # Save predictions
# headlines_df.to_csv('headlines_with_predictions.csv', index=False)
# print(headlines_df[['headline', 'prediction']])
