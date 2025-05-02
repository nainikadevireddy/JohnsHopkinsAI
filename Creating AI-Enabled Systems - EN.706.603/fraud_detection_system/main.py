import json
import datetime
from flask import Flask, request, jsonify
from deployment import DeploymentPipeline
import numpy as np

class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)

@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Welcome to the Fraud Detection System!"


@app.route('/generate_new_dataset', methods=['PUT'])
def generate_new_dataset():
    """
    Route to generate a new dataset.

    URL Params:
    version (str): The version of the dataset.

    Returns:
    JSON response with a success message and description.
    """
    dataset_version = request.args.get('version')
    if not dataset_version:
        return json.dumps({"error": "No version specified"}, cls=json_serialize), 400
    
    try:
        description = pipeline.generate_new_dataset(dataset_version)
        pipeline.log(
            'dataset_descriptions',
            log_entry={
                "version": dataset_version,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "description": description
            },
            log_file=f'dataset_description_{dataset_version}.json'
        )
        return json.dumps({"message": f"Generated new dataset {dataset_version}", "description": description}, cls=json_serialize), 200
    except Exception as e:
        return json.dumps({"error": str(e)}, cls=json_serialize), 500

@app.route('/dataset_description', methods=['GET'])
def get_dataset_description():
    """
    Route to get the description of a dataset version.

    URL Params:
    version (str): The version of the dataset to describe.

    Returns:
    JSON response with the dataset description.
    """
    dataset_version = request.args.get('version')
    if not dataset_version:
        return json.dumps({"error": "No version specified"}, cls=json_serialize), 400
    try:
        description = pipeline.get_log('dataset_descriptions', f'dataset_description_{dataset_version}')
    except FileNotFoundError:
        return json.dumps({"error": "Description not found"},cls=json_serialize), 404
    
    return json.dumps({"description": description}, cls=json_serialize), 200


@app.route('/model_evaluation', methods=['GET'])
def get_model_evaluation():
    """
    Route to get the description of a dataset version.

    URL Params:
    version (str): The version of the dataset to describe.

    Returns:
    JSON response with the dataset description.
    """
    dataset_version = request.args.get('version')
    if not dataset_version:
        return json.dumps({"error": "No version specified"}, cls=json_serialize), 400
    try:
        evaluation = pipeline.get_log('evaluation', f'evaluation_{dataset_version}')
    except FileNotFoundError:
        return json.dumps({"error": "Description not found"}, cls=json_serialize), 404
    
    return json.dumps({"evaluation": evaluation}, cls=json_serialize), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route to make predictions based on input data.

    Request Body:
    JSON object containing input data for prediction.

    Returns:
    JSON response with the prediction result.
    """
    dataset_version = request.args.get('version')
    if not dataset_version:
        return json.dumps({"error": "No version specified"}, cls=json_serialize), 400
    try:

        timestamp = datetime.datetime.now()
        data = request.get_json()
        if not data:
            return json.dumps({"error": "No data provided"}, cls=json_serialize), 400
        
        model_prediction = pipeline.predict(data, dataset_version)
    
        pipeline.log(
            f'predictions/{dataset_version}', 
            log_entry={
                'dataset_version': dataset_version,
                'input_data': data,
                'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': model_prediction
            }, 
            log_file=f'{timestamp.strftime("%Y%m%d_%H%M%S")}.json'
        )
    
        return json.dumps({"prediction": model_prediction}, cls=json_serialize), 200
    except Exception as e:
        return json.dumps({"error": str(e)}, cls=json_serialize), 500

if __name__ == '__main__':
    pipeline = DeploymentPipeline(['transactions_0.csv', 'transactions_1.parquet', 'transactions_2.json'])
    app.run(host='0.0.0.0', port=5000, debug=True)
