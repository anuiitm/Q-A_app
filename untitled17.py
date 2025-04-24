from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import requests
import os
from langchain.schema import Document
import re
import requests
from bs4 import BeautifulSoup

def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        raise ValueError("Unsupported format")
    return loader.load()

def chunk_docs(docs, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    model = "sentence-transformers/all-MiniLM-L6-v2"  # Or use deepseek-ai/embedding if available
    embeddings = HuggingFaceEmbeddings(model_name=model)
    return FAISS.from_documents(chunks, embeddings)

def ask_deepseek(query, docs, api_key):
    context = "\n\n".join([doc.page_content for doc in docs])

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer only using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post("https://api.deepseek.com/v1/chat/completions", json=payload, headers=headers)

    return response.json()["choices"][0]["message"]["content"]
def extract_text_from_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return pytesseract.image_to_string(image)

def extract_text_from_image_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text

def extract_links_from_text(text):
    return re.findall(r'https?://\S+', text)

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""
def handle_structured_query(query, file_path):
    import pandas as pd
    import json
    query_lower = query.lower()

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                return "Unsupported JSON structure."
        else:
            return None

        if "total" in query_lower:
            for col in df.select_dtypes(include="number").columns:
                return f"Total of column '{col}': {df[col].sum()}"

        elif "average" in query_lower:
            for col in df.select_dtypes(include="number").columns:
                return f"Average of column '{col}': {df[col].mean()}"

        elif "maximum" in query_lower:
            for col in df.select_dtypes(include="number").columns:
                return f"Maximum of column '{col}': {df[col].max()}"

        elif "minimum" in query_lower:
            for col in df.select_dtypes(include="number").columns:
                return f"Minimum of column '{col}': {df[col].min()}"

        else:
            return "Structured data detected, but unclear intent."

    except Exception as e:
        return f"Error reading structured file: {str(e)}"
def safe_ask(query, docs, api_key):
    try:
        response = ask_deepseek(query, docs, api_key)
        if not response.strip() or "I don't know" in response.lower():
            return "The information is not available in the provided documents."
        return response
    except Exception as e:
        return f"Error during answer generation: {str(e)}"


deepseek_key = "sk-73d745ef8f1141cba4eeee21a6c34148"
if __name__=='__main__':
    all_chunks = []
    for file in ['iitm_hackathon.docx']:
        docs = load_file(file)
        chunks = chunk_docs(docs)
        all_chunks.extend(chunks)

    vectorstore = create_vectorstore(all_chunks)

    query = "Explain this for a 5th standard student in 500 words atleast."
    relevant_docs = vectorstore.similarity_search(query, k=4)

    # Get answer
    answer = ask_deepseek(query, relevant_docs, deepseek_key)
    print(answer)



    # OCR from image
    img_text = extract_text_from_image("img.png")
    print("Text from image:", img_text)

    # OCR from scanned PDF
    ocr_text = extract_text_from_image_pdf("scanned_doc.pdf")
    print("Text from scanned PDF:", ocr_text)
    docs_from_ocr = [Document(page_content=img_text, metadata={"source": "image"})]
    all_text = "\n".join([doc.page_content for doc in all_chunks])
    urls = extract_links_from_text(all_text)
    ocr_chunks = chunk_docs(docs_from_ocr)
    web_docs = []
    for url in urls:
        html_text = get_text_from_url(url)
        if html_text.strip():
            web_docs.append(Document(page_content=html_text, metadata={"source": url}))
    web_chunks = chunk_docs(web_docs)
    model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model)
    vectorstore = FAISS.from_documents(all_chunks + ocr_chunks + web_chunks, embeddings)
