
# Tweet Generation Project

This project is a **Flask-based API** for generating and retraining tweets using a fine-tuned **GPT-2 model**. The API allows users to upload a dataset for retraining the model and generate tweets based on a prompt.

## Features

- **Retrain Model**: Upload a dataset of tweets to fine-tune the GPT-2 model on your data.
- **Generate Tweets**: Generate tweets with a customizable prompt using the fine-tuned GPT-2 model.

## Requirements

- Python 3.7+
- pip
- Install the required dependencies using the `requirements.txt` file.

## Installation

Follow these steps to set up the project:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/tweet-generation-project.git
cd tweet-generation-project
```

### 2. Create and activate a virtual environment (optional but recommended):

- **On Windows**:

```bash
python -m venv venv
.
env\Scripts ctivate
```

- **On macOS/Linux**:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies:

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### 4. Set up the environment variables by creating a `.env` file:

Create a `.env` file in the project root directory with the following variables:

```bash
MODEL_DIR=./finetuned_model
TRAINING_OUTPUT_DIR=./retrained_results
LOGGING_DIR=./logs
```

### 5. Run the Flask app:

```bash
python run.py  # Or use 'py run.py' if using Windows
```

## Additional Configuration

- Ensure the `.env` file is properly set up to configure training arguments, directories, and other settings.
- If you're facing any security or execution issues with activating the virtual environment, you may need to adjust your system's execution policy for PowerShell. For more information, visit [Execution Policies](https://go.microsoft.com/fwlink/?LinkID=135170).

---

### Notes:

- The application uses **Flask** for the backend API and **GPT-2** for generating tweets.
- Retraining the model will take time based on the dataset size and your hardware configuration.
