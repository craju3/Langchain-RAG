from flask import Flask, request, jsonify
from chatbot import ask_question

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Langchain RAG API!"

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    answer = ask_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
