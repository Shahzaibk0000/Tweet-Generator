from flask import render_template, request, jsonify
from . import app
from .utils import clean_and_format_text, process_data, generate_tweet

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        max_length = int(request.form.get('max_length', 50))
        tweet = generate_tweet(prompt, max_length)
        return jsonify({'generated_tweet': tweet})
    else:
        return render_template('generate.html')


@app.route('/retrain', methods=['GET', 'POST'])
def retrain():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided!'}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed!'}), 400

        try:
            result = process_data(file)
            return jsonify({'message': f'Model retrained successfully! {result}'}), 200
        except Exception as e:
            return jsonify({'error': f'Error during retraining: {str(e)}'}), 500
    else:
        return render_template('retrain.html')


@app.route('/favicon.ico')
def favicon():
    return '', 204
