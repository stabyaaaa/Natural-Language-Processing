from flask import Flask, render_template, request, redirect, url_for
import PyPDF2
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import os

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file.
    """
    text = ''
    with open(pdf_file, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def preprocess_text(text):
    """
    Preprocess the text: remove stop words and punctuation, and convert to lowercase.
    """
    stopwords = list(STOP_WORDS)
    doc = nlp(text)
    clean_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
    return clean_tokens

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'resume' not in request.files:
            return redirect(request.url)
        file = request.files['resume']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)

        if file:
            # Save the uploaded file
            filename = 'uploaded_resume.pdf'
            file.save(filename)
            # Extract text from the PDF
            text = extract_text_from_pdf(filename)
            # Preprocess the text
            cleaned_text = preprocess_text(text)
            # Render the extracted information on the webpage
            return render_template('result.html', text=cleaned_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
