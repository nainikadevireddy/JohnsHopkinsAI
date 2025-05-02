import pandas as pd
import json
import os
from typing import Union, List, Dict, Tuple
import uuid
from datetime import datetime

class FeatureConstructor:
    def __init__(self, raw_data: Union[str, pd.DataFrame], *args, **kwargs):
        self.raw_data = raw_data
        self.dataset = self.extract_data(raw_data)
        self.data_sources = []
        self.version = self.set_version()
        self.creation_date = datetime.now()
        self.args = args
        self.kwargs = kwargs

    def extract_data(self, dat: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(dat, pd.DataFrame):
            return dat
        elif isinstance(dat, str):
            file_extension = os.path.splitext(dat)[1]
            if file_extension == '.csv':
                data = pd.read_csv(dat)
            elif file_extension == '.parquet':
                data = pd.read_parquet(dat)
            elif file_extension == '.json':
                data = pd.read_json(dat)
            else:
                raise ValueError("Unsupported file format")
            self.data_sources.append(dat)
            return data
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

    def get_data_sources(self) -> List[str]:
        return self.data_sources

    def set_version(self) -> str:
        return str(uuid.uuid4())

    def transform(self, *args, **kwargs) -> pd.DataFrame:
        return self.dataset.transform(*args, **kwargs)

    def describe(self, dataframe: pd.DataFrame, output_file: str = None) -> Dict:
        description = {
            'version': self.version,
            'data sources': self.get_data_sources(),
            'column names': list(dataframe.columns),
            'date ranges': self.get_date_ranges(dataframe)
        }
        measures = self.get_quality_measures(dataframe)
        result = {
            'description': description,
            'measures': measures
        }

        if output_file:
            name, ext = os.path.splitext(filename)
            date_str = self.creation_date.strftime('%Y%m%d')
            versioned_filename = f"{name}_v{self.version}_{date_str}{ext}"
            with open(versioned_filename, 'w') as f:
                json.dump(result, f, indent=4)

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
