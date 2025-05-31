def clear_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Alexey3250/Start_ML/refs/heads/main/Machine_learning/10_classification/banking.csv', delimiter = ',')

    ### Уберем колонку duration

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
