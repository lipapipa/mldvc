import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import json

def evaluate(model_path, test_data_path, metrics_path):
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)
    
    X = df.drop('y', axis=1)
    y = df['y']
    
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y, preds),
        'f1_score': f1_score(y, preds),
        'roc_auc': roc_auc_score(y, probas),
        'confusion_matrix': confusion_matrix(y, preds).tolist()
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-data", type=str, required=True)
    parser.add_argument("--metrics", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.test_data, args.metrics)
