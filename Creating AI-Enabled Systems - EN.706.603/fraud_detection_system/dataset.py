import pandas as pd
import json
import os
from typing import Union, List, Dict, Tuple
import uuid
from datetime import datetime
from data_engineering import DataEngineering
import random
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle

class DatasetConstructor:
    def __init__(self, raw_data: Union[str, List[str], pd.DataFrame], *args, **kwargs):
        self.version = self.set_version()
        self.raw_data = raw_data
        self.data_sources = []
        self.creation_date = datetime.now()
        self.data = self.extract_data(raw_data)
        self.processed_data = self.process_data(self.data)
        self.raw_data_description = self.describe(self.data, version=self.version)
        self.processed_data_description = self.describe(self.processed_data, version=self.version)
        self.args = args
        self.kwargs = kwargs
        self.format = 'json'

    def extract_data(self, dat: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(dat, pd.DataFrame):
            return dat
        elif isinstance(dat, str):
            dat = [dat]
        
        if isinstance(dat, list):
            data_frames = []
            for file_path in dat:
                file_extension = os.path.splitext(file_path)[1]
                if file_extension == '.csv':
                    df = pd.read_csv(file_path)
                elif file_extension == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_extension == '.json':
                    df = pd.read_json(file_path)
                else:
                    raise ValueError("Unsupported file format")
                self.data_sources.append(file_path)
                data_frames.append(df)
            combined_data = pd.concat(data_frames, ignore_index=True)
            combined_data = combined_data.set_index('Unnamed: 0')
            combined_data.index.name = None
            return combined_data
        else:
            raise ValueError("Unsupported data type")

    def load(self, output_filename: str, format: str = 'csv') -> None:
        name, ext = os.path.splitext(filename)
        versioned_filename = f"{name}_v{self.version}{ext}"
        
        if format == 'csv':
            self.dataset.to_csv(output_filename, index=False)
        elif format == 'parquet':
            self.dataset.to_parquet(output_filename, index=False)
        elif format == 'json':
            self.dataset.to_json(output_filename, orient='records', lines=True)
        else:
            raise ValueError("Unsupported file format")

    def get_data_source(self) -> List[str]:
        return self.data_sources

    def set_version(self) -> str:
        return str(uuid.uuid4())

    def sample(self, *args, **kwargs) -> pd.DataFrame:
        return self.dataset.sample(*args, **kwargs)

    def describe(self, dataframe: pd.DataFrame,version, output_file: str = None) -> Dict:
        description = {
            'version': version,
            'data sources': self.get_data_source(),
            'column names': list(dataframe.columns),
            'date ranges': self.get_date_ranges(dataframe)
        }
        measures = self.get_quality_measures(dataframe)
        result = {
            'description': description,
            'measures': measures
        }

        return result

    def get_date_ranges(self, dataframe: pd.DataFrame) -> Tuple[datetime, datetime]:
        date_columns = dataframe.select_dtypes(include=['datetime', 'datetime64']).columns
        if not date_columns.any():
            return (None, None)
        min_date = dataframe[date_columns].min().min()
        max_date = dataframe[date_columns].max().max()
        return (min_date, max_date)

    def get_quality_measures(self, dataframe: pd.DataFrame) -> Dict:
        measures = {
            'total_rows': dataframe.shape[0],
            'total_columns': dataframe.shape[1],
            'missing_values': dataframe.isnull().sum().sum(),
            'missing_value_percentage': (dataframe.isnull().sum().sum() / dataframe.size) * 100,
            'column_missing_values': dataframe.isnull().sum().to_dict(),
            'duplicate_rows': dataframe.duplicated().sum(),
            'unique_values': dataframe.nunique().to_dict(),
            'data_types': dataframe.dtypes.astype(str).to_dict(),
            'mean': dataframe.mean(numeric_only=True).to_dict(),
            'median': dataframe.median(numeric_only=True).to_dict(),
            'std_dev': dataframe.std(numeric_only=True).to_dict(),
            'min': dataframe.min(numeric_only=True).to_dict(),
            'max': dataframe.max(numeric_only=True).to_dict(),
            'zero_variance_columns': [col for col in dataframe.columns if dataframe[col].nunique() == 1]
        }
        return measures

    def process_data(self, data) -> Dict:
        #data = data[data['is_fraud'].notna()]
        processed_data = DataEngineering(data)
        processed_data.remove_duplicates()
        processed_data.dataset = processed_data.dataset[processed_data.dataset['is_fraud'].notna()]
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

        processed_data.dataset['amt'] = processed_data.dataset['amt'].replace(np.nan, processed_data.dataset['amt'].median()) 
        processed_data.dataset['amt'].fillna(processed_data.dataset['amt'].median(), inplace=True)
        processed_data.dataset['day_of_week'].fillna(processed_data.dataset['day_of_week'].mode()[0], inplace=True)
        processed_data.dataset['hour_of_day'].fillna(processed_data.dataset['hour_of_day'].mode()[0], inplace=True)
        processed_data.dataset['category'].fillna(processed_data.dataset['category'].mode()[0], inplace=True)
        processed_data.dataset['city_pop'].fillna(processed_data.dataset['city_pop'].median(), inplace=True)
        processed_data.dataset['age_group'].fillna(processed_data.dataset['age_group'].mode()[0], inplace=True)
        processed_data.dataset['distance'].fillna(processed_data.dataset['distance'].median(), inplace=True)

        
        processed_data.dataset = processed_data.dataset[['amt', 'day_of_week', 'hour_of_day', 'category', 'sex', 'city_pop', 'age_group', 'distance', 'is_fraud']]


        #one hot encoding that is saved to file to use in inference data
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_data = enc.fit_transform(processed_data.dataset[['category']])
        feature_names = enc.get_feature_names_out(['category'])
        enc_df = pd.DataFrame(enc_data.toarray(), columns=feature_names)
    
        # Reset indices before concatenating
        processed_data.dataset = processed_data.dataset.reset_index(drop=True)
        enc_df = enc_df.reset_index(drop=True)
    
        # Concatenate the dataset with the one-hot encoded columns
        dataset = pd.concat([processed_data.dataset[['amt', 'day_of_week', 'hour_of_day', 'sex', 'city_pop', 'age_group', 'distance']], enc_df, processed_data.dataset[['is_fraud']]], axis=1)


        with open('encoder.pickle', 'wb') as f:
            pickle.dump(enc, f)
                             
        dataset = dataset.apply(pd.to_numeric, errors='coerce')
        dataset = dataset[[x for x in dataset if x not in 'is_fraud'] + ['is_fraud']]
        
        return dataset

    def generate_dataset(self, dataset_version, label_column = 'is_fraud'):

        X = self.processed_data.drop(columns=[label_column])
        Y = self.processed_data[label_column]

        if Y.isnull().any():
            X = X[Y.notna()]
            Y = Y[Y.notna()]
        
        undersample = RandomUnderSampler(sampling_strategy=0.1)
        
        X_under, Y_under = undersample.fit_resample(X, Y)

        oversample = RandomOverSampler()
        
        X_over, Y_over = oversample.fit_resample(X_under, Y_under)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_over)

        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_df[label_column] = Y_over.values
        
        self.load(dataset_version, X_scaled_df)

        description = self.describe(X_scaled_df, version=dataset_version)

        return description
    
    def load(self, dataset_version, dataset_df):
        """
        Load the dataset based on the specified version and file format.

        Returns:
        pd.DataFrame: The loaded dataset.
        """

        os.makedirs('resources/datasets', exist_ok=True)
        file_path = f"resources/datasets/{dataset_version}.{self.format}"

        if self.format == 'csv':
            return dataset_df.to_csv(file_path, index=False)
        elif self.format == 'parquet':
            return dataset_df.to_parquet(file_path)
        elif self.format == 'json':
            return dataset_df.to_json(file_path, orient='records', lines=True)
