"""data loading & preprocessing funcs"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    return df

def train_test_split_time_aware(df, test_size=0.2, random_state=42):
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)