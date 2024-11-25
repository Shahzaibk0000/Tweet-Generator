import re
import pandas as pd
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from . import app
from .config import TRAINING_ARGS
from sklearn.model_selection import train_test_split
from transformers import Trainer, DataCollatorForLanguageModeling

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

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding=True, truncation=True, max_length=256)

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
from flask import jsonify
from . import app

# Define a function outside the process_data function for tokenization
def tokenize_function_example(examples, tokenizer):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def process_data(file):
    try:
        # Load CSV and clean text
        df = pd.read_csv(file, header=None)
        df.columns = ['tweet_text']
        df['tweet_text'] = df['tweet_text'].fillna("").apply(clean_and_format_text)

        # Split into training and evaluation sets
        train_texts, eval_texts = train_test_split(df['tweet_text'].tolist(), test_size=0.1, random_state=42)
        train_dataset = Dataset.from_dict({'text': train_texts})
        eval_dataset = Dataset.from_dict({'text': eval_texts})

        # Tokenize datasets
        tokenized_train = train_dataset.map(lambda x: tokenize_function_example(x, app.config['tokenizer']), batched=True)
        tokenized_eval = eval_dataset.map(lambda x: tokenize_function_example(x, app.config['tokenizer']), batched=True)

        # Remove raw text columns and set PyTorch format
        tokenized_train = tokenized_train.remove_columns(["text"])
        tokenized_eval = tokenized_eval.remove_columns(["text"])
        tokenized_train.set_format("torch")
        tokenized_eval.set_format("torch")

        # Initialize Trainer
        trainer = Trainer(
            model=app.config['model'],  # Load the current model from app config
            args=TrainingArguments(
                output_dir='./app/retrained_results',
                num_train_epochs=3,  # Increase epochs for better learning
                per_device_train_batch_size=8,
                save_steps=500,
                save_total_limit=2,
                logging_dir='./app/logs',
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=300,
                learning_rate=5e-5,
                gradient_accumulation_steps=4,
                fp16=True,  # Enable mixed precision training if GPU is available
                report_to="none",  # Disable WandB or other tracking tools
            ),
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=DataCollatorForLanguageModeling(tokenizer=app.config['tokenizer'], mlm=False),
        )

        # Train the model
        print("Starting training...")
        try:
            train_output = trainer.train()
            app.config['model'].save_pretrained('./app/model/finetuned_model')
            app.config['tokenizer'].save_pretrained('./app/model/finetuned_model')

            # Extract training metrics for response
            metrics = train_output.metrics
            return jsonify({
                'message': 'Model retrained successfully!',
                'training_loss': metrics.get('train_loss', 'N/A'),
                'eval_loss': metrics.get('eval_loss', 'N/A'),
                'training_steps': metrics.get('global_step', 'N/A'),
            }), 200

        except Exception as e:
            print(f"Training error: {str(e)}")
            return jsonify({'error': f"Error during training: {str(e)}"}), 500

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({'error': f"Error during retraining: {str(e)}"}), 500



def generate_tweet(prompt, max_length):
    model = app.config['model']
    tokenizer = app.config['tokenizer']

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    tweet = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    generated_tweet = tweet[0]['generated_text'].strip()
    formatted_tweet = generated_tweet.replace("<TWEET_START>", "").replace("<TWEET_END>", "").strip()
    formatted_tweet = formatted_tweet[:280]
    if not formatted_tweet.endswith("#Crypto #Blockchain"):
        formatted_tweet += " #Crypto #Blockchain"
    return formatted_tweet
