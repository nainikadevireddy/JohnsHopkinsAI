# Visual Search System

## Description
The Visual Identification System for IronClad Technology is designed to verify the identity of personnel using facial recognition. The system processes image data, extracts embeddings, searches for matches, and dynamically manages access control. It is implemented in Python with Flask for the API and Docker support for easy deployment.

## Modules

### 1. Image Capture and Preprocessing
- **Preprocessing**: Handles resizing of images to ensure uniformity and facilitates efficient processing for the deep learning model.

### 2. Deep Learning Model
- **Model**: Utilizes ResNet-based architectures (ResNet-18 and ResNet-34) trained with SimCLR for self-supervised learning to extract high-dimensional feature embeddings.

### 3. Embedding Storage and Search
- **KDTree**: Efficient storage and retrieval of embeddings using KDTree for fast nearest neighbor search.

### 4. Access Control Management
- **Dynamic Management**: Methods to add new personnel by generating and storing embeddings and to remove departing personnel by deleting their embeddings and associated images.

### 5. Flask Application
- Provides endpoints for image preprocessing, model selection, adding/removing personnel, and searching the gallery.

## Contents
These are the files you can find in this repository.
- **`main.py`**: The main application script that starts the Flask server and defines the API routes.
- **`pipeline.py`**: Contains the `Pipeline` class, which handles the deployment logic for ML models.
- **`model.py`**: Contains the `Model` class, which defines the model architecture and prediction logic.
- **`preprocess.py`**: Contains the `Preprocessing` class for image preprocessing
- **`search.py `**: Contains the KDTreeSearch class for searching the embeddings.
- **`indexing.py`**: Contains the KDTree class for indexing the embeddings.
- **`metrics.py`**: Contains the `Metrics` class, which calculates evaluation metrics for the model. 
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`Dockerfile`**: Defines the Docker image configuration for the project.


## Running the Application
### Instructions to Run the Flask Application locally

Here are step-by-step instructions to run the Flask application locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/devireddy-nainika/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/devireddy_nainika.git
   cd visual_search_system
   ```

2. **Create the Requirements File:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run Flask application:**

   Run the Flask application

   ```sh
   python main.py
   ```

3. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:5000
   ```

   You should see the "Welcome to the Visual Search System" message from the index route. Your flask server is now properly running!

4. **Using the API Endpoints:**

    **Supported Endpoints**
    - **GET /**: Returns a welcome message.
    - **POST /authenticate**: Selects the model and precomputes the tree if not already saved. Requires image_size and resnet_variant parameters.
    - **PUT /add_identity**: Adds a new identity to the gallery. Requires image_path and name parameters.
    - **DELETE /remove_identity**: Removes an identity from the gallery. Requires name parameter.
    - **POST /search_gallery**: Searches for the nearest neighbors in the gallery. Requires a probe image.


    **Example Requests**
    - **Authenticate Model:**

        ```sh
        curl -X POST "http://localhost:5000/authenticate?image_size=224&resnet_model=resnet_018"
        ```
        Note: Only accepts 224 or 64 as the image size. Only accepts resnet_018 and resnet_034 for the resnet_model.
    
    - **Add Identity::**

        ```sh
        curl -X PUT "http://localhost:5000/add_identity?image_path=path_to_image.jpg&name=John_Doe"
        ```
        Note: Replace spaces with '_' in name.


    - **Remove Identity:**

        ```sh
        curl -X DELETE "http://localhost:5000/add_identity?name=John_Doe"
        ```
        Note: Replace spaces with '_' in name.


    - **Search Gallery:**

        ```sh
        curl -X POST "http://localhost:5000/search" -F "file=@/path/to/probe_image.jpg" -F "k=5"
        ```

    - **Access Logs:**

        ```sh
        curl -X GET "http://localhost:5000/access_logs?start_time=2024-01-01%2000:00:00&end_time=2024-01-31%2023:59:59"
        ```

### Instructions to Run the Flask Application with Docker:
It is highly recommended that you develop locally before moving to docker. Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t visual_search_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it -v $(pwd)/resources:/app/resources -p 5000:5000 visual_search_system:latest
   ```

