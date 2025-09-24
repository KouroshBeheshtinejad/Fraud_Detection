import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

class FraudModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.smote = SMOTE()

    def fit(self, X, y):
        X_res, y_res = self.smote.fit_resample(X, y)
        self.model.fit(X_res, y_res)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]

    def save(self, path='models/rf_model.joblib'):
        joblib.dump(self.model, path)
        print(f"Model saved to: {path}")

    def load(self, path='models/rf_model.joblib'):
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")
