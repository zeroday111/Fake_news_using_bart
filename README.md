# Fake_news_using_bart
# Fake News Detection with BERT

This repository contains code for training a BERT-based model to classify news headlines as fake or real.

## Overview

This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for sequence classification. The model is trained on a dataset of news headlines labeled as fake or real.

## Technologies Used

- **Python**: Programming language used for implementation
- **PyTorch**: Deep learning framework for building and training neural networks
- **Transformers Library**: Hugging Face library for pre-trained Transformer models like BERT
- **Pandas**: Library for data manipulation and analysis
- **scikit-learn**: Library for machine learning tasks (metrics used for evaluation)

## Setup Instructions

1. **Install Dependencies**:
   - Install Python dependencies using `pip install -r requirements.txt`

2. **Download Pre-trained BERT Model**:
   - The model uses `bert-base-uncased` from Hugging Face Transformers. It will be automatically downloaded when initialized.

3. **Dataset**:
   - Ensure the dataset (`combined_news_dataset.csv`) is placed in the correct directory (`/content` in this example).

## Training

- **Tokenization**: News headlines are tokenized using BERT's tokenizer.
- **Model**: BERT model is fine-tuned for sequence classification (fake vs real news).
- **Optimizer**: AdamW optimizer with a linear scheduler is used for training.

## Evaluation

- Metrics such as accuracy, precision, recall, and F1 score are calculated after training.

## Usage

- Use the `test_model` function to classify new text inputs after training.

```python
new_text = "tomorrow world will fall agents the china"
prediction = test_model(new_text, model, tokenizer, max_length)
print(f'Prediction for "{new_text}": {prediction}')


# News Headline Classification with BERT

This project aims to classify news headlines as either 'Fake News' or 'Real News' using a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. The workflow includes training the model on a labeled dataset, evaluating its performance, and using it to classify new headlines collected from the NewsAPI.

## Prerequisites
- Python 3.6 or higher
- A [NewsAPI](https://newsapi.org/) API Key

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/news-headline-classification.git
   cd news-headline-classification
