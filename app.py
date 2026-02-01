from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import uuid
import traceback

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

# === Setup ===
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Global Vars ===
db = None
qa_chain = None
embeddings = None

# === Initialize Embeddings Once ===
def init_embeddings():
    global embeddings
    if embeddings is None:
        print("[DEBUG] Loading HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("[DEBUG] Embeddings loaded successfully")
    return embeddings

# === Routes ===

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_pdf():
    global db, qa_chain

    file = request.files.get('pdf')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        filename = str(uuid.uuid4()) + '.pdf'
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(f"[DEBUG] File saved to {file_path}")

        # === Load and Chunk PDF ===
        print("[DEBUG] Loading PDF...")
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        print(f"[DEBUG] Loaded {len(documents)} pages from PDF")

        if len(documents) == 0:
            return jsonify({"error": "PDF is empty or unreadable"}), 400

        print("[DEBUG] Splitting documents...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        print(f"[DEBUG] Created {len(docs)} chunks")

        # === Embeddings (HuggingFace) ===
        print("[DEBUG] Initializing embeddings...")
        embeddings = init_embeddings()
        print("[DEBUG] Creating FAISS index...")
        db = FAISS.from_documents(docs, embeddings)
        print("[DEBUG] FAISS index created successfully")

        # === Create QA Chain ===
        print("[DEBUG] Creating QA chain...")
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template("""Based on the context provided, answer the question concisely:

Context: {context}

Question: {input}

Answer:""")

        retriever = db.as_retriever()
        
        # Simple retrieval chain without LLM - just return the relevant documents
        qa_chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "input": RunnablePassthrough()
            }
            | prompt
            | RunnableLambda(lambda x: x.to_string())  # Just format the prompt, don't call LLM
        )

        print("[DEBUG] QA chain created successfully")
        return jsonify({"message": "PDF processed and chat is ready!", "filename": filename}), 200
    
    except Exception as e:
        print(f"[ERROR] Exception during upload: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    global qa_chain, db, embeddings
    if not db:
        return jsonify({'response': 'Please upload a PDF first.'}), 400

    query = request.json.get('message', '').strip()
    if not query:
        return jsonify({'response': 'Empty question'}), 400

    try:
        print("[QUERY RECEIVED]", query)
        
        # Simple semantic search without LLM
        retriever = db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        
        if relevant_docs:
            answer = "\n\n".join([doc.page_content for doc in relevant_docs])
        else:
            answer = "No relevant information found in the document."
        
        print("[RESPONSE]", answer[:100])
        return jsonify({'response': answer}), 200
    except Exception as e:
        print("[ERROR] Exception during chat:", str(e))
        traceback.print_exc()
        return jsonify({'response': f'[Error] {str(e)}'}), 500


if __name__ == '__main__':
    print("[INFO] Starting Flask app...")
    app.run(debug=True)