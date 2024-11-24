import re
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from . import app

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

def process_data(request):
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        df = pd.read_csv(file, header=None)
        df.columns = ['tweet_text']
        df['tweet_text'] = df['tweet_text'].fillna("").apply(clean_and_format_text)

        train_texts, eval_texts = train_test_split(df['tweet_text'].tolist(), test_size=0.1, random_state=42)
        train_dataset = Dataset.from_dict({'text': train_texts})
        eval_dataset = Dataset.from_dict({'text': eval_texts})

        tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        tokenized_eval = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        tokenized_train = tokenized_train.remove_columns(["text"])
        tokenized_eval = tokenized_eval.remove_columns(["text"])
        tokenized_train.set_format("torch")
        tokenized_eval.set_format("torch")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )

        trainer.train()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        return jsonify({'message': 'Model retrained successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
