import numpy as np 

def add_simple_features(X):
    X = X.copy()
    if 'Amount' in X.columns:
        X['Amount_log'] = np.log1p(X['Amount'])
        X['Amount_by_mean'] = X['Amount'] / (X['Amount'].mean() + 1e-9)
    return X