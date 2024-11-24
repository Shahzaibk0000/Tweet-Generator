from flask import Flask
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .config import MODEL_DIR
import os

app = Flask(__name__)

model_dir = MODEL_DIR if os.path.exists(MODEL_DIR) else None

if model_dir is None:
    print("Local model not found. Loading pretrained GPT-2 model from Hugging Face.")
    model_dir = 'gpt2'


tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
model = GPT2LMHeadModel.from_pretrained(model_dir)

app.config['model'] = model
app.config['tokenizer'] = tokenizer

from . import routes
