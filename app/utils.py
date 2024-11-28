import re
import pandas as pd
from datasets import Dataset
from transformers import (
    pipeline,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    GPT2LMHeadModel
)
from flask import jsonify
from sklearn.model_selection import train_test_split
import language_tool_python
import hashlib
import logging

from app.config import get_training_args
from . import app

logging.basicConfig(level=logging.INFO)
cache = {}
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

trainer = None

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=256)

def clean_and_format_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    crypto_keywords = ['bitcoin', 'ethereum', 'blockchain', 'crypto', 'nft', 'web3', 'token']
    hashtags = ' '.join(f'#{kw}' for kw in crypto_keywords[:3])
    formatted_tweet = f"{text.strip()} {hashtags}"
    return formatted_tweet[:280]


def process_data(request_or_file):
    global trainer
    try:
        if hasattr(request_or_file, 'files'):
            if 'file' not in request_or_file.files:
                return jsonify({'error': 'No file uploaded'}), 400

            file = request_or_file.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
        else:
            file = request_or_file
        df = pd.read_csv(file, header=None)
        df.columns = ['tweet_text']
        df['tweet_text'] = df['tweet_text'].fillna("").apply(clean_and_format_text)

        train_texts, eval_texts = train_test_split(df['tweet_text'].tolist(), test_size=0.1, random_state=42)
        train_dataset = Dataset.from_dict({'text': train_texts})
        eval_dataset = Dataset.from_dict({'text': eval_texts})

        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

        tokenized_train = tokenized_train.remove_columns(["text"]).set_format("torch")
        tokenized_eval = tokenized_eval.remove_columns(["text"]).set_format("torch")

        trainer = Trainer(
            model=model,
            args=get_training_args(),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )


        logging.info("Starting training...")
        train_output = trainer.train()

        app.config['model'].save_pretrained('app/model/finetuned_model')
        app.config['tokenizer'].save_pretrained('app/model/finetuned_model')

        metrics = train_output.metrics
        return jsonify({
            'message': 'Model retrained successfully!',
            'training_loss': metrics.get('train_loss', 'N/A'),
            'eval_loss': metrics.get('eval_loss', 'N/A'),
            'training_steps': metrics.get('global_step', 'N/A'),
        }), 200

    except Exception as e:
        logging.error(f"Error during retraining: {e}")
        return jsonify({'error': f"Error during retraining: {str(e)}"}), 500

# Generate a tweet
def generate_tweet(prompt, max_length):
    model = app.config['model']
    tokenizer = app.config['tokenizer']

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    tweet = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.27,
    )
    
    generated_tweet = tweet[0]['generated_text'].strip()
    formatted_tweet = generated_tweet.replace("<TWEET_START>", "").replace("<TWEET_END>", "").strip()
    formatted_tweet = formatted_tweet[:280]

    if not formatted_tweet.endswith("#Crypto #Blockchain"):
        formatted_tweet += " #Crypto #Blockchain"
    
    return formatted_tweet

def correct_grammar(text, language='en-US'):
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    if text_hash in cache:
        return cache[text_hash]

    tool = language_tool_python.LanguageTool(language)
    tool.disabled_rules = ['STYLE', 'SPELLING']

    try:
        chunk_size = 1000
        if len(text) > chunk_size:
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            corrected_text = ''.join([language_tool_python.utils.correct(chunk, tool.check(chunk)) for chunk in chunks])
        else:
            corrected_text = language_tool_python.utils.correct(text, tool.check(text))

        cache[text_hash] = corrected_text
        logging.info(f"Grammar corrected successfully for text of length {len(text)}")
        return corrected_text
    except Exception as e:
        logging.error(f"Error correcting grammar: {e}")
        return text
