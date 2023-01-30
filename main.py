# https://github.com/venky14/Machine-Learning-with-Iris-Dataset/blob/master/Machine%20Learning%20with%20Iris%20Dataset.ipynb
# Importing all dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class Hyperparameters(object):
    filepath: str = "Iris.csv"
    test_size: float = 0.25
    random_state: int = 6

hp = Hyperparameters()

# Collecting and preparing data
def create_dataframe(filepath):
    df = pd.read_csv(filepath)
    df.drop("Id", axis=1, inplace = True)
    return df


# Splitting train and test dataset
def split_dataset(df, test_size, random_state):
    X = df.drop(["Species"], axis=1)
    y = df["Species"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Building and fitting the model
def train_model(X_train, y_train):
    model = SVC()
    return model.fit(X_train, y_train)


# Running the workflow
def run_wf(hp: Hyperparameters) -> SVC:
    df = create_dataframe(filepath=hp.filepath)
    X_train, X_test, y_train, y_test = split_dataset(df=df, test_size=hp.test_size, random_state=hp.random_state)
    return train_model(X_train=X_train, y_train=y_train)

if __name__=="__main__":
    run_wf(hp=hp)