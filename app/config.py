import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', 'app/model/finetuned_model')
TRAINING_OUTPUT_DIR = os.getenv('TRAINING_OUTPUT_DIR', 'app/retrained_results')
LOGGING_DIR = os.getenv('LOGGING_DIR', 'app/logs')

# Configure LanguageTool for grammar checking
GRAMMAR_TOOL_URL = os.getenv('GRAMMAR_TOOL_URL', 'http://localhost:8010/v2/check')

from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir='./retrained_results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=5000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=50,
        eval_strategy="no",
        eval_steps=500,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        fp16=True,
    )

