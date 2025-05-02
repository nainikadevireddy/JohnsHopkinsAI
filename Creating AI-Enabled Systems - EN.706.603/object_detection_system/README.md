## Docker Instructions

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).


2. **Build the Docker Image**

To build the Docker image, run the following command:

```sh
docker build -t object-detection-system:latest .
 ```

3. **Run the Docker Container**

To run the Docker container, use the following command:

```sh
docker run --it -p 23000:23000/udp object-detection-system:latest
 ```

# Fraud Detection System

## Description
The Object Detection System is designed to identify significant objects within warehouse environments using machine learning. The system captures video frames from a live feed, processes the frames to detect objects, and provides interfaces for human reviewers to correct detection errors. The system is implemented in Python and Flask, with Docker support for easy deployment.


## Modules

### 1. Video Stream Capture
- **UDP Video Stream Source:** Captures video frames from a live feed at 30 frames per second.

### 2. Object Detection
- **YOLOv4-tiny Object Detector:**  Detects significant objects in each frame using a pre-trained YOLOv4-tiny model.

### 3. Non-Maximal Suppression (NMS)
- **NMS Module**: Filters overlapping bounding boxes to retain the most accurate detections.

### 4. Inference on CPU
- **Inference Engine:** Performs object detection inference using CPU resources.

### 5. Hard Negative Mining
- **Hard Negative Miner:** Identifies and augments difficult examples to improve model robustness.


## Contents
These are the files you can find in this repository.
- **`main.py`**: The main application script that starts the Flask server and defines the API routes.
- **`pipeline.py`**: Contains the InferenceService class, which handles the inference logic for the object detection model.
- **`video_processing.py`**: Contains the VideoProcessing class, which handles video stream capture and frame processing.
- **`object_detection.py`**: Contains the YOLOObjectDetector class, which defines the object detection model.
- **`non_maximal_suppression.py`**: Contains the NMS class, which implements the non-maximal suppression algorithm.
- **`hard_negative_mining`**.py: Contains the HardNegativeMiner class, which handles hard negative mining.
- **`metrics.py`**: Contains the Metrics and Loss classes, which calculate evaluation metrics for the model.
- **`requirements.txt`**: Lists the Python dependencies required for the project.
- **`Dockerfile`**: Defines the Docker image configuration for the project.

## Running the Application
### Instructions to Run the Flask Application locally

Here are step-by-step instructions to run the Flask application locally:

1. **Clone the Repository:**

   Clone this [repository](https://github.com/creating-ai-enabled-systems-summer-2024/devireddy-nainika/tree/main).

   ```sh
   git clone https://github.com/creating-ai-enabled-systems-summer-2024/devireddy_nainika.git
   cd object_detection_system
   ```

2. **Install the Dependencies:**

   Install all necessary Python libraries:

   ```sh
   pip install -r requirements.txt
   ```

3. **Initiate UDP Stream:**

   Open two terminals. In the first terminal, run the UDP stream client using ffplay:
   ```sh
   ffplay udp://127.0.0.1:23000
   ```

   In the second terminal, run the UDP stream server using ffmpeg:
    ```sh
   ffmpeg -re -i ./yolo_resources/test_videos/worker-zone-detection.mp4 -r 30 -vcodec mpeg4 -f mpegts udp://127.0.0.1:23000

   ```
    
4. **Run Flask application:**

   Run the Flask application

   ```sh
   python main.py
   ```

5. **Access the Flask Application:**

   Open your web browser and navigate to:

   ```
   http://localhost:5001
   ```

   You should see the "Welcome to the Object Detection System" message from the index route. Your flask server is now properly running!

6. **Using the API Endpoints:**

    **Supported Endpoints**
    - **GET /**: Returns a welcome message.
    - **POST /process_video**: Processes the video stream and performs object detection.
    - **GET /list_detections**: Lists detections within a specified frame range.
    - **GET /images_with_detections**: Retrieves images with their corresponding detections.
    - **GET /top_hard_negatives**: Identifies and augments the top-N hard negatives.
    - **GET /detections/<filename>**: Retrieves specific detection files.

    **Example Requests**
    - **Process Video:**

        ```sh
        curl -X POST "http://localhost:5001/process_video?format=YOLO"
        ```
    
    - **List Detections:**

        ```sh
        curl -X GET "http://localhost:5001/list_detections?frame_range=0-100"
        ```
        Note: Only implements frame ranges of two numerical values separated by a dash. (eg. 0-100)

    - **Images with Detections:**

        ```sh
        curl -X GET "http://localhost:5001/images_with_detections?frame_range=0-100&format=YOLO"
        ```
        Note: Only implements frame ranges of two numerical values separated by a dash. (eg. 0-100)

    - **Top Hard Negatives:**

        ```sh
        curl -X GET "http://localhost:5001/top_hard_negatives?top_n=10"
        ```
    - **Get Detection File:**

        ```sh
        curl -X GET "http://localhost:5001/detections/<filename>"
        ```

### Instructions to Run the Flask Application with Docker:
It is highly recommended that you develop locally before moving to docker. Here are step-by-step instructions to run the Flask application with Docker:

1. **Install Docker:**

   Make sure Docker is installed on your machine. You can download and install Docker from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. **Build the Docker Image:**

   Build the Docker image from the Dockerfile:

   ```sh
   docker build -t object_detection_system:latest .
   ```

3. **Run the Docker Container:**

   Run the Docker container with the built image:

   ```sh
   docker run -it -v $(pwd)/resources:/app/resources -p 5001:5001 object_detection_system:latest
   ```

## Quick Start

### Interacting with the System

1. **Process the Video Stream**:
    ```sh
    curl -X POST "http://localhost:5001/process_video?format=YOLO"
    ```

2. **List Detections**:
    ```sh
    curl -X GET "http://localhost:5001/list_detections?frame_range=0-100"
    ```
    
3. **Get Top Hard Negatives:**:
    ```sh
    curl -X GET "http://localhost:5001/top_hard_negatives?top_n=10"
    ```
   
