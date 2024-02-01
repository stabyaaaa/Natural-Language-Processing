from flask import Flask, render_template, request, jsonify
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load model from local path
model_path = 'model'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained('model/tokenizer')
# Set the model to evaluation mode
model.eval()

# Adjustable parameters for optimization
MAX_SEQ_LEN = 100
TEMPERATURE = 0.8
BATCH_SIZE = 5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        prompt = request.form['input_text']
        generated_text = generate_text(prompt)
        return jsonify({'generated_text': generated_text})

def generate_text(prompt):
    tokens = tokenizer.encode(prompt, return_tensors='pt')

    # Check if tokens is empty
    if tokens.numel() == 0:
        return "Unable to generate text. Please provide a valid prompt."

    try:
        # Generate text dynamically as the user types using batching
        for _ in range(MAX_SEQ_LEN // BATCH_SIZE):
            output = model.generate(tokens, max_length=len(tokens[0])+BATCH_SIZE, do_sample=True, temperature=TEMPERATURE, num_beams=5, pad_token_id=tokenizer.eos_token_id)
            new_tokens = output[:, -BATCH_SIZE:]
            tokens = torch.cat((tokens, new_tokens), dim=1)

        generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        error_message = f"Error during generation: {e}"
        print(error_message)
        return error_message

if __name__ == '__main__':
    app.run(debug=True)
