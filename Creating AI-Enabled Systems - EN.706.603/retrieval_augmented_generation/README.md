# Retrieval-Augmented Generation (RAG) System

## Description
The Retrieval-Augmented Generation (RAG) System is designed to combine document retrieval with generative answering. It allows users to search a corpus of documents, retrieve the top-K relevant contexts, and generate answers based on both the question and retrieved information. The system is implemented in Python and Flask, with Docker support for easy deployment.

## Modules

### 1. Document Processing and Preprocessing
- **DocumentProcessing**: Handles cleaning and preprocessing of input documents before embedding.
- **Embedding**: Extracts vector embeddings from documents and queries for similarity search.

### 2. Retrieval Pipeline
- **Pipeline**: Orchestrates search and generation workflows, manages the document corpus, and integrates embedding search with generative answering.

### 3. Flask Application
- Provides endpoints for searching contexts, asking questions with generated answers, adding new documents, and removing documents from the corpus.

## Contents
These are the files you can find in this repository.
- **`main.py`**: The main application script that starts the Flask server and defines the API routes.
- **`pipeline.py`**: Contains the `Pipeline` class, which manages search, generation, and document updates.
- **`embedding.py`**: Contains the `Embedding` class, which handles embedding extraction.
- **`preprocessing.py`**: Contains the `DocumentProcessing` class, which processes documents before embedding.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`Dockerfile`**: Defines the Docker image configuration for the project.

## Running the Application
### Instructions to Run the Flask Application locally

Here are step-by-step instructions to run the Flask application locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/devireddy-nainika/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/devireddy_nainika.git
   cd rag_system
   ```

2. **Install the Dependencies:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run Flask application:**

   Run the Flask application

   ```sh
   python main.py
   ```

4. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:5000
   ```

   You should see the "Welcome to the Fraud Detection System" message from the index route. Your flask server is now properly running!

4. **Using the API Endpoints:**

    **Supported Endpoints**

    - **POST /search**: Returns top-K matching contexts for a given question.
    - **POST /ask**: Generates an answer and returns the supporting context.
    - **POST /add_document**: Adds a new document to the corpus (requires file path).
    - **POST /remove_document**: Removes a document from the corpus (by name).

    **Example Requests**
    - **Search:**

        ```sh
        curl -X POST "http://localhost:5000/search" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "k": 5}'
        ```
    
    - **Ask:**

        ```sh
        curl -X POST "http://localhost:5000/ask" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "k": 5}'
        ```
  

    - **Add Document:**

        ```sh
        curl -X POST "http://localhost:5000/add_document" -H "Content-Type: application/json" -d '{"file_path": "/path/to/myfile.txt"}'
        ```


    - **Remove Document:**

        ```sh
        curl -X POST "http://localhost:5000/remove_document" -H "Content-Type: application/json" -d '{"document_name": "myfile.txt"}'
        ```


### Instructions to Run the Flask Application with Docker:
It is highly recommended that you develop locally before moving to docker. Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t retrieval_augmented_analysis:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it -v $(pwd)/resources:/app/resources -p 5000:5000 retrieval_augmented_analysis:latest
   ```

## Quick Start

### Interacting with the System

1. **Search for Top-K Contexts**:
    ```sh
    curl -X POST "http://localhost:5000/search" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "k": 5}'
    ```

2. **Generate an Answer**:
    ```sh
    curl -X POST "http://localhost:5000/ask" -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "k": 5}'
    ```
    
3. **Add a New Document**:
    ```sh
    curl -X POST "http://localhost:5000/add_document" -H "Content-Type: application/json" -d '{"file_path": "/path/to/myfile.txt"}'
    ```
    
4. **Remove a Document**:
    ```sh
    curl -X POST "http://localhost:5000/add_document" -H "Content-Type: application/json" -d '{"file_path": "/path/to/myfile.txt"}'
    ```
