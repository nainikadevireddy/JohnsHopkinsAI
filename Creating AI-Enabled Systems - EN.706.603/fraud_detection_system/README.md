# Fraud Detection System

## Description
The Fraud Detection System is designed to identify fraudulent transactions using machine learning. The system processes transaction data, trains a model to detect fraud, and provides an interface for making predictions on new transactions. The system is implemented in Python and Flask, with Docker support for easy deployment.

## Modules

### 1. Data Ingestion and Processing
- **DatasetConstructor**: Handles the extraction, processing, sampling and versioning of datasets.
- **DataEngineering**: Provides functions for data preprocessing such as removing duplicates, handling missing values, standardizing dates, and feature engineering.

### 2. Model Training and Evaluation
- **Model**: Manages the training and saving of machine learning models using scikit-learn.
- **Metrics**: Computes evaluation metrics for trained models, including accuracy, precision, recall, F1 score, and ROC-AUC score.

### 3. Deployment Pipeline
- **DeploymentPipeline**: Orchestrates the data processing, model training, evaluation, and prediction workflows.

### 4. Flask Application
- Provides endpoints for generating new datasets, describing datasets, and making predictions.


## Contents
These are the files you can find in this repository.
- **`main.py`**: The main application script that starts the Flask server and defines the API routes.
- **`deployment_pipeline.py`**: Contains the `DeploymentPipeline` class, which handles the deployment logic for ML models.
- **`dataset.py`**: Contains the `DatasetConstructor` class, which handles dataset generation and loading.
- **`model.py`**: Contains the `Model` class, which defines the model architecture and prediction logic.
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
   cd fraud_detection_system
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

   You should see the "Welcome to the Fraud Detection System" message from the index route. Your flask server is now properly running!

4. **Using the API Endpoints:**

    **Supported Endpoints**
    - **GET /**: Returns a welcome message.
    - **PUT /generate_new_dataset**: Generates a new dataset. Requires a `version` parameter.
    - **GET /dataset_description**: Retrieves the description of a dataset. Requires a `version` parameter.
    - **POST /predict**: Makes a prediction based on input data.

    **Example Requests**
    - **Generate Dataset:**

        ```sh
        curl -X PUT "http://localhost:5000/generate_new_dataset?version=<DATASET_VERSION>"
        ```
        Note: Only implements `DATASET_VERSION` as a string.
    
    - **Get Dataset Description:**

        ```sh
        curl -X GET "http://localhost:5000/dataset_description?version=<DATASET_VERSION>"
        ```
        Note: Only implements `DATASET_VERSION` that have been previously created by the user using generate_new_dataset.

    - **Get Model Evaluation:**

        ```sh
        curl -X GET "http://localhost:5000/model_evaluation?version=<DATASET_VERSION>"
        ```
        Note: Only implements the model trained on the `DATASET_VERSION` that have been previously created by the user using generate_new_dataset.

    - **Make a Prediction:**

        ```sh
        curl -X POST "http://localhost:5000/predict?version=<DATASET_VERSION>" -H "Content-Type: application/json" -d @<LOG_FILE>
        ```
        For example, you can run `curl -X POST "http://localhost:5000/predict?version=datasetA" -H "Content-Type: application/json" -d @test_data.json`
        Note: Only implements the model of the specified `DATASET_VERSION` that have been previously created by the user using generate_new_dataset.



### Instructions to Run the Flask Application with Docker:
It is highly recommended that you develop locally before moving to docker. Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t fraud_detection_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it -v $(pwd)/resources:/app/resources -p 5000:5000 fraud_detection_system:latest
   ```

## Quick Start

### Interacting with the System

1. **Generate a new dataset**:
    ```sh
    curl -X PUT "http://localhost:5000/generate_new_dataset?version=datasetA"
    ```

2. **Get dataset description**:
    ```sh
    curl -X GET "http://localhost:5000/dataset_description?version=datasetA"
    ```
    
3. **Get model evaluation**:
    ```sh
    curl -X GET "http://localhost:5000/model_evaluation?version=datasetA"
    ```
    
4. **Make a prediction**:
    ```sh
    curl -X POST "http://localhost:5000/predict?version=datasetA" -H "Content-Type: application/json" -d @test_data.json
    ```

### Sample `test_data.json`
Ensure your `test_data.json` file is formatted correctly and contains the necessary fields:
```json
{
    "trans_date_trans_time": "2021-01-01 12:34:56",
    "cc_num": 1234567890123456,
    "merchant": "XYZ Store",
    "category": "grocery",
    "amt": 100.0,
    "first": "John",
    "last": "Doe",
    "sex": "M",
    "street": "123 Main St",
    "city": "Anytown",
    "state": "NY",
    "zip": "12345",
    "lat": 40.7128,
    "long": -74.0060,
    "city_pop": 100000,
    "job": "engineer",
    "dob": "1980-01-01",
    "trans_num": "1234abcd",
    "unix_time": 1609459200,
    "merch_lat": 40.7159,
    "merch_long": -74.0033
}
