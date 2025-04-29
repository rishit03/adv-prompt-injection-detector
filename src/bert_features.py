# src/bert_features.py

from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

# Load BERT model and tokenizer (you can also try distilBERT later for faster speed)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Put model into evaluation mode
model.eval()

@torch.no_grad()
def encode_prompts(prompts):
    """
    Encodes a list of prompts into BERT embeddings.
    
    Args:
        prompts (list of str): The text prompts to encode.

    Returns:
        np.ndarray: The embeddings matrix.
    """
    embeddings = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        # Take the [CLS] token (first token) as sentence embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)

    return np.vstack(embeddings)
