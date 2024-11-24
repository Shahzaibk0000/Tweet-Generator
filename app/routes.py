from flask import request, jsonify
from . import app
from .utils import clean_and_format_text, process_data, generate_tweet

@app.route('/')
def home():
    return "Welcome to the Tweet Generation API!"

@app.route('/retrain', methods=['POST'])
def retrain():
    return process_data(request)

@app.route('/generate', methods=['GET'])
def generate():
    prompt = request.args.get('prompt', '')
    max_length = int(request.args.get('max_length', 50))
    tweet = generate_tweet(prompt, max_length)
    return jsonify({'generated_tweet': tweet}), 200


@app.route('/favicon.ico')
def favicon():
    return '', 204