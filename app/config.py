import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', 'app/model/finetuned_model')
TRAINING_OUTPUT_DIR = os.getenv('TRAINING_OUTPUT_DIR', 'app/retrained_results')
LOGGING_DIR = os.getenv('LOGGING_DIR', 'app/logs')

# Configure LanguageTool for grammar checking
GRAMMAR_TOOL_URL = os.getenv('GRAMMAR_TOOL_URL', 'http://localhost:8010/v2/check')

TRAINING_ARGS = {
    'output_dir': './retrained_results',
    'num_train_epochs': 3,  # Increased epochs for better fine-tuning
    'per_device_train_batch_size': 8,
    'save_steps': 5000,
    'save_total_limit': 2,
    'logging_dir': './logs',
    'logging_steps': 50,
    'evaluation_strategy': "steps",
    'eval_steps': 500,
    'gradient_accumulation_steps': 2,
    'learning_rate': 3e-5,  # Fine-tuned learning rate
    'fp16': True,
    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1.0,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'report_to': "none",
    'seed': 42,
}
