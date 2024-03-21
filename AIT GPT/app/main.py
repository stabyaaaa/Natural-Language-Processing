import os
from flask import Flask, render_template, request

app = Flask(__name__)

chat_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form.get("prompt")
        answer = get_response(prompt)
        chat_history.append({'prompt': prompt, 'answer': answer})
    return render_template('index.html', chat_history=chat_history)

def get_response(prompt):
    # Your logic to generate a response goes here
    return "This is a placeholder response."

if __name__ == '__main__':
    app.run(debug=True)
