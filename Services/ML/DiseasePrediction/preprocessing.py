import pandas as pd
from sklearn.model_selection import train_test_split
def preprocessing(data):
    df=pd.read_csv(data)
    print(df.isnull().sum())
    df.fillna(0)
    X=df.drop('target',axis=1)
    Y=df['target']
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=42)
    return X_train,X_test,Y_train,Y_test