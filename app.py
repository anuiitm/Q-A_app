from flask import Flask, request, render_template, redirect, jsonify,flash, get_flashed_messages
from werkzeug.utils import secure_filename
import os
from untitled17 import load_file, chunk_docs, create_vectorstore, safe_ask, handle_structured_query
from langchain.schema import Document
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re
import markdown
app = Flask(__name__)
app.secret_key = 'super-secret-key' 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

all_chunks = []
vectorstore = None
uploaded_files = []
file_answers = {}

# OCR for images and PDFs
def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(image)

def extract_text_from_image_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

# Crawl text from URLs in documents
def extract_links_from_text(text):
    return re.findall(r'https?://\S+', text)

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        return ""
documents_uploaded=False
@app.route('/')
def index():
    if documents_uploaded:
        return render_template('index.html', answer=None, error="")
    return render_template('index.html', answer=None, error=None)


@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore, uploaded_files, all_chunks

    uploaded_files = request.files.getlist("files")
    all_chunks = []

    for f in uploaded_files:
        filename = secure_filename(f.filename)
        file_path = os.path.join("uploads", filename)
        f.save(file_path)

        try:
            docs = load_file(file_path)
            chunks = chunk_docs(docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    if all_chunks:
        vectorstore = create_vectorstore(all_chunks)
        flash("✅ Files uploaded successfully. You can now ask questions.", "success")
    else:
        flash("⚠️ No valid files were uploaded. Please try again.", "danger")

    return redirect("/")



@app.route("/ask", methods=["POST"])
def ask():
    global vectorstore
    question = request.form.get("question")

    if not vectorstore:
        return render_template("index.html", answer={}, error="Please upload documents first.")   
    deepseek_key = "sk-73d745ef8f1141cba4eeee21a6c34148"
    docs = vectorstore.similarity_search(question, k=100000000)
    raw_answer = safe_ask(question, docs, deepseek_key)
    rendered_answer = markdown.markdown(raw_answer)
    return render_template("index.html", answer={"All Files": rendered_answer}, error="")
if __name__ == '__main__':
    app.run(debug=True)
