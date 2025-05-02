from flask import Flask, request, jsonify
from src.extraction.embedding import Embedding
from src.extraction.preprocessing import DocumentProcessing

app = Flask(__name__)

# Initialize the pipeline
pipeline = Pipeline()

@main.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    question = data.get('question')
    k = data.get('k', 5)
    top_contexts = pipeline.search_context(question, k)
    return jsonify({"contexts": top_contexts}), 200

@main.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    k = data.get('k', 5)

    context = pipeline.search_context(question, 5)
    
    # Generate the answer using the pipeline
    answer = pipeline.generate_answer(question, k)
    
    return jsonify({"question": question, "answer": answer, "context": context}), 200

@main.route('/add_document', methods=['POST'])
def add_document():
    file_path = request.json.get('file_path')
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found."}), 404
    
    # Define the destination path in storage/corpus
    destination_path = os.path.join("storage/corpus", os.path.basename(file_path))
    
    # Copy the document to the corpus directory
    shutil.copy(file_path, destination_path)
    
    # Process the new document
    pipeline.add_document(destination_path)

    return jsonify({"message": f"Document {file_path} added to the corpus."}), 200

@main.route('/remove_document', methods=['POST'])
def remove_document():
    document_name = request.json.get('document_name')
    
    # Define the file path in storage/corpus
    file_path = os.path.join("storage/corpus", document_name)
    
    if not os.path.exists(file_path):
        return jsonify({"error": f"Document {document_name} not found."}), 404
    
    # Remove the document from the corpus
    pipeline.remove_document(document_name)
    
    # Delete the document from the storage/corpus directory
    os.remove(file_path)
    
    return jsonify({"message": f"Document {document_name} removed from the corpus."}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    
    pipeline.precompute()
