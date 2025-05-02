import pandas as pd
from datetime import datetime

class DataEngineering: 
    def __init__(self, data):
        #initialize dataset
        self.dataset = data

    def load_dataset(self, filename):
        try: 
            #load dataset if available
            data = pd.read_csv(filename, index_col=0)
            return data
        except Exception as e: 
            raise Exception(f"Error loading dataset: {e}")

    def describe(self, N: int):
        #total number of rows
        print(f"Total number of rows: {len(self.dataset)}")

        #type of columns
        print(f"Column types:\n{self.dataset.dtypes}")

        #print first n rows
        return self.dataset.head(N)

    def clean_missing_values(self):
        # Fill missing values with the median for numerical columns and mode for categorical columns
        for column in self.dataset.columns:
            if self.dataset[column].isnull().sum() > 0:
                if self.dataset[column].dtype == 'object':
                    # Fill missing categorical values with the mode
                    if not self.dataset[column].mode().empty:
                        self.dataset[column].fillna(self.dataset[column].mode()[0], inplace=True)
                else:
                    # Fill missing numerical values with the median
                    self.dataset[column].fillna(self.dataset[column].median(), inplace=True)

    def remove_duplicates(self):
        #remove duplicated values
        self.dataset.drop_duplicates(inplace=True)

    def remove_blank_labels(self, column_name):
        self.dataset = self.dataset[self.dataset[column_name].notna()]

    def capitalize(self):
        for column in self.dataset.columns:
            if self.dataset[column].dtype == 'object':
                self.dataset[column] = self.dataset[column].str.upper() 

    def standardize_dates(self, column_name):
        #standardize and correct inconsistencies in date formats
        self.dataset[column_name] = pd.to_datetime(self.dataset[column_name], errors='coerce')

    def trim_spaces(self, column_name):
        #trim spaces from strings
        self.dataset[column_name] = self.dataset[column_name].str.strip()

    def resolve_anomalous_dates(self, column_name):
        #remove dates that are in the future
        self.dataset = self.dataset[self.dataset[column_name] <= datetime.now()]

    def expand_dates(self, column_name):
        #append 'day_of_week' and 'hour_of_day' columns derived from the transaction to facilitate time-based analysis
        self.dataset['day_of_week'] = self.dataset[column_name].apply(lambda x: x.weekday() if not pd.isnull(x) else np.nan)
        self.dataset['hour_of_day'] = self.dataset[column_name].apply(lambda x: x.hour if not pd.isnull(x) else np.nan)
        self.dataset[column_name] = self.dataset[column_name].dt.strftime('%m/%d/%Y %H:%M:%S')

    def categorize_transactions(self, low, medium, high):
        #Categorize transaction amounts into "Low" (bottom 25%), "Medium" (25% to 75%), and "High" (above 75%) based on quantile ranges
        quantiles = self.dataset['amt'].quantile([low, medium, high]).tolist()
        bins = [self.dataset['amt'].min(), quantiles[0], quantiles[1], quantiles[2]]
        labels = ['Low', 'Medium', 'High']
        self.dataset['transaction_category'] = pd.cut(self.dataset['amt'], bins=bins, labels=labels, include_lowest=True)

    def range_checks(self):
        #implement range checks to validate that numerical columns post-transformation fall within expected bounds
        #let's assume the amount should be between 0 and 100,000
        if not self.dataset['amt'].between(0, 100000).all():
            raise ValueError("Range check failed for 'amt' column")

    def null_checks(self):
        #ensure there are no null values in essential columns after cleaning and transformations
        if self.dataset.isnull().any().any():
            raise ValueError("Null values found after cleaning")

    def type_validation(self):
        #confirm that all data types are consistent with expected formats post-cleanup, particularly ensuring that numerical data is not inadvertently stored as strings
        expected_types = {
            'amt': 'float64',
            'trans_date_trans_time': 'datetime64[ns]',
            'dob': 'datetime64[ns]', 
            'hour_of_day': 'int32', 
            'day_of_week': 'object',
            'transaction_category': 'category'
        }
        for column, expected_type in expected_types.items():
            if self.dataset[column].dtype != expected_type:
                raise TypeError(f"Column {column} does not match expected type {expected_type}")

    def uniqueness_validation(self):
        #ensure that all transactions are unique where expected, and check for unintentional duplication after data transformation processes
        if self.dataset.duplicated().any():
            raise ValueError("Duplicate transactions found")

    def historical_data_consistency(self):
        #check that new data entries are consistent with historical trends or benchmarks in transaction volumes or amounts
        historical_avg = 50  # This would come from historical data
        current_avg = self.dataset['amt'].mean()
        if not (0.5 * historical_avg <= current_avg <= 1.5 * historical_avg):
            raise ValueError("Current data is inconsistent with historical averages")

    def categorical_data_validation(self):
        #verify that all entries in categorical fields match an approved list of categories, especially after transformations or data merging activities
        approved_categories = ['MISC_NET', 'GROCERY_POS', 'SHOPPING_POS', 'GAS_TRANSPORT', 'GROCERY_NET', 'MISC_POS', 'SHOPPING_NET', 'FOOD_DINING', 'ENTERTAINMENT', 'KIDS_PETS', 'HOME', 'TRAVEL', 'HEALTH_FITNESS', 'PERSONAL_CARE']  
        if not self.dataset['category'].str.upper().isin(approved_categories).all():
            raise ValueError("Categorical data validation failed for 'category'")
