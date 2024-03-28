from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and tokenizer
model_path = "model/instruction_tuning"
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define instruction prompt functionnn
def instruction_prompt(instruction, prompt_input=None):
    if prompt_input:
        return f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {prompt_input}

        ### Response:
        """.strip()
    else:
        return f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        """.strip()

# Define route for home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        instruction = request.form['instruction']
        prompt_input = request.form['prompt_input']
        output = text_generator(instruction_prompt(instruction, prompt_input))
        generated_text = output[0]['generated_text'].split("### Response:\n")[-1].strip()
        return render_template('result.html', instruction=instruction, prompt_input=prompt_input, generated_text=generated_text)
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
