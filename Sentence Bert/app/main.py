from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

# Load pre-trained BERT model and tokenizer
model = torch.load('models/pbert.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_seq_length = 128

# Function to calculate similarity
def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenizer(sentence_a, return_tensors='pt', max_length=max_seq_length, truncation=True, padding='max_length').to(device)
    inputs_b = tokenizer(sentence_b, return_tensors='pt', max_length=max_seq_length, truncation=True, padding='max_length').to(device)

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids']
    attention_a = inputs_a['attention_mask']
    inputs_ids_b = inputs_b['input_ids']
    attention_b = inputs_b['attention_mask']
    segment_ids = torch.zeros(1, max_seq_length, dtype=torch.int32).to(device)

    # Extract token embeddings from BERT
    u = model.get_last_hidden_state(inputs_ids_a, segment_ids)  # all token embeddings A = batch_size, seq_len, hidden_dim
    v = model.get_last_hidden_state(inputs_ids_b, segment_ids)  # all token embeddings B = batch_size, seq_len, hidden_dim

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score

@app.route('/index.html', methods=['POST'])
def get_similarity():
    data = request.json
    sentence_a = data['sentence_a']
    sentence_b = data['sentence_b']
    similarity = calculate_similarity(model, tokenizer, sentence_a, sentence_b, device)
    return jsonify({'similarity_score': similarity})

if __name__ == '__main__':
    app.run(debug=True)
