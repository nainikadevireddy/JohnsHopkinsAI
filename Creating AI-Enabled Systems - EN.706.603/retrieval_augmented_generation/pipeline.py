from src.extraction.embedding import Embedding
from src.extraction.preprocessing import DocumentProcessing
from src.generator.question_answering import BERTQuestionAnswer
from src.retrieval.index import KDTree
from src.retrieval.search import KDTreeSearch, Measure
from glob import glob
import pandas as pd
import os
import numpy as np

class Pipeline:
    
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', qa_model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
        self.embedding_model = Embedding(embedding_model_name)
        self.qa_model = BERTQuestionAnswer(qa_model_name)
        self.kd_tree = None

    def __predict(self, text_chunk: str) -> np.ndarray:
        return self.embedding_model.encode(text_chunk)

    def __save_embeddings(self, embedding: np.ndarray, document_filename: str, segment_number: int) -> None:
        # Define the directory and filename
        directory = "storage/embeddings/"
        filename = f"{document_filename}_{segment_number}.npy"
        
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the embedding to the file
        np.save(os.path.join(directory, filename), embedding)

    def precompute(self): 
        documents = glob("storage/corpus/*.txt.clean")
        processing = DocumentProcessing()
        embeddings = []
        metadata_list = []
        
        for document in documents:
            document_filename = os.path.basename(document).split('.')[0]
            chunks = processing.split_document(document, sentences_per_chunk=3)

            for idx, chunk in enumerate(chunks):
                embedding = self.__predict(chunk)
                self.__save_embeddings(embedding, document_filename, idx)
                embeddings.append(embedding)
                metadata_list.append({"text": chunk, "document": document_filename, "segment": idx})

        self.kd_tree = KDTree(k=len(embeddings[0]), points=embeddings, metadata_list=metadata_list)

    def search_context(self, question: str, k: int = 5) -> list:
        question_embedding = self.__predict(question)
        kd_search = KDTreeSearch(self.kd_tree, Measure.euclidean)
        nearest_neighbors = kd_search.find_nearest_neighbors(question_embedding, k)
        return [neighbor[1] for neighbor in nearest_neighbors]

    def add_document(self, file_path):
        processing = DocumentProcessing()
        document_filename = os.path.basename(file_path).split('.')[0]
        chunks = processing.split_document(file_path, sentences_per_chunk=3)

        for idx, chunk in enumerate(chunks):
            embedding = self.__predict(chunk)
            self.__save_embeddings(embedding, document_filename, idx)
            self.kd_tree.insert(embedding, metadata={"text": chunk, "document": document_filename, "segment": idx})

    def remove_document(self, document_name):
        self.embeddings = [e for e, m in zip(self.embeddings, self.metadata_list) if m["document"] != document_name]
        self.metadata_list = [m for m in self.metadata_list if m["document"] != document_name]
        self.kd_tree = KDTree(k=len(self.embeddings[0]), points=self.embeddings, metadata_list=self.metadata_list)

    def generate_answer(self, question: str, k: int = 5) -> str:
        # Step 1: Retrieve the top k contexts
        top_contexts = self.search_context(question, k)
        
        # Extract the text from the top contexts
        context_texts = [context['text'] for context in top_contexts]

        # Step 2: Use the BERT model to generate an answer based on the question and contexts
        answer = self.qa_model.get_answer(question, context_texts)
        
        return answer
        
if __name__ == "__main__":
    # Initialize the pipeline and precompute embeddings
    pipeline = Pipeline()
    pipeline.precompute()

    # Load the questions from qa_resources/questions.csv
    questions = pd.read_csv("qa_resources/questions.csv", delimiter="\t")
    
    # Iterate through the questions and search for the top context
    for index, row in questions.iterrows():
        question = row['Question']
        print(f"\nQuestion: {question}")
        
        # Search for the top contexts
        top_contexts = pipeline.search_context(question, k=5)
        
        # Print the top contexts
        print("Top Contexts:")
        for context in top_contexts:
            print(f"Document: {context['document']}, Segment: {context['segment']}")
            print(f"Text: {context['text']}\n")


