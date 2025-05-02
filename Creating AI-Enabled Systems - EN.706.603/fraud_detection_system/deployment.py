import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import DatasetConstructor
from model import Model
from metrics import Metrics


class json_serialize(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class DeploymentPipeline:
    """
    DeploymentPipeline class handles the deployment logic for ML models,
    including training and inference pipelines.
    """

    def __init__(self, data_sources):
        """
        Initialize the DeploymentPipeline with necessary components.
        """
        self.data = DatasetConstructor(data_sources)
        self.log('raw_data', self.data.raw_data_description, f'description_{self.data.version}.csv')
        self.log('processed_data', self.data.processed_data_description, f'description_{self.data.version}.csv')

    def train_and_evaluate(self, dataset_version, label_column='is_fraud'):
        """
        Train and evaluate the model using the specified dataset version and label column.

        Parameters:
        dataset_version (str): The version of the dataset to use for training.
        label_column (str): The name of the label column in the dataset.

        Returns:
        dict: Evaluation metrics of the trained model.
        """


        file_path = f"resources/datasets/{dataset_version}.json"
        dataset = pd.read_json(file_path, lines=True)
        
        # Split the dataset into training and testing sets
        X = dataset.drop(columns=[label_column])
        y = dataset[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        model = Model(dataset_version)       

        # Train the model
        model.train(X_train, y_train)
        
        # Evaluate the model
        metrics = Metrics(model, X_test, y_test)
        evaluation_results = metrics.run_metrics()
        
        # Log the evaluation results
        self.log('evaluation', evaluation_results, f'evaluation_{dataset_version}.json')
        
        return evaluation_results

    def predict(self, data, data_version):
        """
        Make predictions using the deployed model.

        Parameters:
        data (dict): Input data for prediction.

        Returns:
        dict: Prediction results.
        """

        model = Model(data_version)
        model.set_model(data_version)

        processed_data = model.preprocess_input(data)
        
        prediction = model.predict(processed_data)

        return prediction

    def get_log(self, component, reference):
        """
        Get the description of a component (e.g., dataset, model, etc.) from the logs.

        Parameters:
        component (str): The component name.
        reference (str): The reference ID for the logs.

        Returns:
        dict: Description from the logs.
        """
        log_path = f"resources/logs/{component}/{reference}.json"
        try:
            with open(log_path, "r") as logs:
                description = json.load(logs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {log_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from log file: {log_path}")

        return description

    def generate_new_dataset(self, dataset_version):
        """
        Generate a new dataset based on the provided version.

        Parameters:
        dataset_version (str): The version of the dataset to generate.

        Returns:
        DataFrame: The generated dataset.
        """

        description = self.data.generate_dataset(dataset_version)
        self.log('dataset_descriptions', description, f'dataset_description_{dataset_version}.json')

        self.train_and_evaluate(dataset_version)
        
        return description
        
    def log(self, component, log_entry, log_file):
        """
        Log the details to a specified log file.

        Parameters:
        component (str): The component name (e.g., dataset, model, etc.).
        log_entry (dict): The log entry to save.
        log_file (str): The name of the log file.
        """
        os.makedirs(f'resources/logs/{component}', exist_ok=True)
        with open(f'resources/logs/{component}/{log_file}', "w") as log:
            json.dump(log_entry, log, indent=4, cls=json_serialize)
        
