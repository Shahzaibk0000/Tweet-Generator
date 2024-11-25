import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = os.getenv('MODEL_DIR', 'app/model/finetuned_model')
TRAINING_OUTPUT_DIR = os.getenv('TRAINING_OUTPUT_DIR', 'app/retrained_results')
LOGGING_DIR = os.getenv('LOGGING_DIR', 'app/logs')

TRAINING_ARGS = {
    'output_dir': './retrained_results',
    'num_train_epochs': 3,
    'per_device_train_batch_size': 8,
    'save_steps': 1000,
    'save_total_limit': 3,
    'logging_dir': './logs',
    'logging_steps': 20,
    'evaluation_strategy': "steps",
    'eval_steps': 500,
    'gradient_accumulation_steps': 4,
    'learning_rate': 5e-5,
    'warmup_steps': 500,
    'warmup_ratio': 0.1,
    'weight_decay': 0.01,
    'fp16': True,
    'adam_beta1': 0.9,
    'adam_beta2': 0.98,
    'adam_epsilon': 1e-8,
    'max_grad_norm': 1.0,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'greater_is_better': False,
    'report_to': ['tensorboard'],
    'disable_tqdm': False,
    'push_to_hub': False,
    'local_rank': -1,
    'seed': 42,
    'report_to': "none",
}
