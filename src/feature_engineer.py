
import pandas as pd
from sklearn.preprocessing import StandardScaler

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Масштабирование числовых признаков
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = numeric_cols.drop('y', errors='ignore')
    
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    engineer_features(args.input, args.output)
