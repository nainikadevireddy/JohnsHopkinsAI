import os
import json
import random
import joblib
from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from typing import Dict
from data_engineering import DataEngineering
import pickle
from datetime import datetime
import random
import numpy as np
from math import radians, sin, cos, sqrt, atan2


class Model:
    """
    Model class that implement model behaviors including 
    versioning, training, and making predictions.
    """

    def __init__(self, dataset_version, model_dir='resources/models', model_name='random_forest'):
        """
        Initialize the Model with a specified architecture.

        """
        self.model_dir = model_dir
        self.model_name = model_name
        self.version = dataset_version
        self.model_path = os.path.join(self.model_dir, f"{self.model_name}_{dataset_version}.pkl")
        self.model = None

    def set_model(self, data_version):
        """
        Load a pre-trained model from a specified path.

        Parameters:
        """
        path = os.path.join(self.model_dir, f"{self.model_name}_{data_version}.pkl")
        if os.path.exists(path):
            self.model = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def train(self, X_train, y_train):
        """
        Train the model with provided training data.

        Parameters:
        X_train (DataFrame or ndarray): Training features.
        y_train (Series or ndarray): Training labels.
        """
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)
        self.save_model()

    def save_model(self):
        """
        Save the trained model to the specified path.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def preprocess_input(self, data):
        """
        Preprocess the input data to match the model's expected format.

        Parameters:
        data (dict): Input data for prediction.

        Returns:
        DataFrame: Preprocessed data in a DataFrame.
        """
        # Convert the data dictionary to a DataFrame
        columns=['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt', 'first', 'last', 'sex', 'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'merch_lat', 'merch_long']

        df = pd.DataFrame([data], columns=columns)

        processed_data = DataEngineering(df)
        processed_data.capitalize()
        processed_data.standardize_dates('trans_date_trans_time')
        processed_data.resolve_anomalous_dates('trans_date_trans_time')
        processed_data.expand_dates('trans_date_trans_time')

        processed_data.dataset['sex'] = processed_data.dataset['sex'].replace({'M': 0, 'F': 1})
        processed_data.dataset['sex'].fillna(random.choice([0, 1]), inplace=True)

        processed_data.standardize_dates('dob')
        processed_data.resolve_anomalous_dates('dob')
        today = datetime.today()
        processed_data.dataset['age'] = processed_data.dataset['dob'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
        bins = [0, 18, 30, 40, 50, 60, 70, 80, 100]
        labels = [0, 1, 2, 3, 4, 5, 6, 7 ]
        processed_data.dataset['age_group'] = pd.cut(processed_data.dataset['age'], bins=bins, labels=labels, right=False)

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in kilometers
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            distance = R * c
            return distance
        
        processed_data.dataset['distance'] = processed_data.dataset.apply(lambda row: haversine(row['lat'], row['long'], row['merch_lat'], row['merch_long']), axis=1)
        dataset = processed_data.dataset[['amt', 'day_of_week', 'hour_of_day', 'category', 'sex', 'city_pop', 'age_group', 'distance']]

        with open('encoder.pickle', 'rb') as f:
            enc = pickle.load(f)

        enc_data = enc.transform(dataset[['category']])    
        inference_feature_names = enc.get_feature_names_out(['category'])
        inference_encoded_df = pd.DataFrame(enc_data.toarray(), 
                                    columns=inference_feature_names)

        dataset = dataset.reset_index(drop=True)
        inference_encoded_df = inference_encoded_df.reset_index(drop=True)
        
        final_df = pd.concat([dataset[['amt', 'day_of_week', 'hour_of_day', 'sex', 'city_pop', 'age_group', 'distance']], 
                                inference_encoded_df], axis=1)
            
        final_df = final_df.apply(pd.to_numeric, errors='coerce')
        
        return final_df

    def predict(self, data):
        """
        Make predictions based on input data.

        Parameters:
        data (dict): Input data for prediction.

        Returns:
        int: Prediction result.
        """
        
        # Make prediction
        prediction = self.model.predict(data)
        return prediction
        
