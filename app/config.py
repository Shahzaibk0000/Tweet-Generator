import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', 'app/model/finetuned_model')
TRAINING_OUTPUT_DIR = os.getenv('TRAINING_OUTPUT_DIR', 'app/retrained_results')
LOGGING_DIR = os.getenv('LOGGING_DIR', 'app/logs')

TRAINING_ARGS = {
    'output_dir': './retrained_results',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 8,
    'save_steps': 1500,
    'save_total_limit': 2,
    'logging_dir': './logs',
    'logging_steps': 10,
    'evaluation_strategy': "steps",
    'eval_steps': 300,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'warmup_steps': 100,
    'fp16': True
}
