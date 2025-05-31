import pandas as pd

def clean_data(input_path, output_path):
    df = pd.read_csv(input_path)
    df = df.dropna().drop_duplicates()  # Удаляем пропуски и дубликаты
    df = df.drop('duration', axis=1)
    ### Посмотрим на категориальные колонки

    categorical_columns = df.loc[:,df.dtypes=='object'].columns

    df = df.drop(['loan', 'housing', 'marital'], axis=1)
    categorical_columns = categorical_columns.drop(['loan', 'housing', 'marital'])

    for col in categorical_columns:
    
        ### К колонкам с маленькой размерностью применим one-hot
        if df[col].nunique() < 5:
            one_hot = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat((df.drop(col, axis=1), one_hot), axis=1)
            
        ### К остальным - счетчики
        else:
            mean_target = df.groupby(col)['y'].mean()
            df[col] = df[col].map(mean_target)
    
    categorical_columns = df.loc[:,df.dtypes=='object'].columns
    
    df = df.reset_index(drop=True) 
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    clean_data(args.input, args.output)
