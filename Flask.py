from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('summarization')
tokenizer = T5Tokenizer.from_pretrained('tokenizer')

# Function to summarize text
def summarize(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define a route for the summarization API
@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()

        # Check if 'text' is in the JSON data
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']

        # Check if the text is empty
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        # Summarize the text
        summary = summarize(text)
        return jsonify({'summary': summary})

    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
