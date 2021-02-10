import pandas as pd
import torch
from torch.utils.data import Dataset

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

import numpy as np

def load_preprocessed_titanic_dataset(limit_len=-1, test_size=0.2):
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')
    df_label = pd.read_csv('./data/gender_submission.csv')
    df_test['Survived'] = df_label['Survived']

    df = pd.concat([df_train, df_test])
    df = preprocess_titanic_dataset(df)

    if limit_len > 0 and limit_len * 1.25 < int(df.__len__()):
        df = df.sample(int(np.ceil(limit_len * 1.25)))

    inp = df.iloc[:, 1:].values
    oup = df.iloc[:, 0].values.reshape(df.__len__(), 1)
    X_train, X_test, y_train, y_test = train_test_split(inp, oup, test_size=test_size)

    return  X_train, X_test, y_train, y_test



class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, input_array, y=None):
        return self

    def transform(self, input_array, y=None):
        return input_array


class TitanicDataset(Dataset):
    def __init__(self, csvpath=None, csv_label_path=None, mode='train', limit_len=-1, X=None, y=None):
        self.mode = mode

        if csvpath != None:
            df = pd.read_csv(csvpath)
            if mode != 'train' and csv_label_path:
                df_label = pd.read_csv(csv_label_path)
                df['Survived'] = df_label['Survived']

            df = preprocess_titanic_dataset(df)
            if limit_len > 0 and limit_len < df.__len__():
                df = df.sample(limit_len)

            self.inp = df.iloc[:, 1:].values
            self.oup = df.iloc[:, 0].values.reshape(df.__len__(), 1)
        else:
            self.inp = X
            self.oup = y


    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inpt = torch.Tensor(self.inp[idx])
        outpt = torch.Tensor(self.oup[idx])

        return {
            'inp': inpt,
            'oup': outpt
        }


def preprocess_titanic_dataset(df):
    df["Pclass"] = df["Pclass"].apply(str)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer()),
        ('normalizer', RobustScaler())
    ])

    processor = ColumnTransformer(
        transformers=[
            ('identity', IdentityTransformer(), ["Survived"]),
            ('transformer', categorical_transformer, ["Pclass", "Sex", "Embarked"]),
            ('normalizer', continuous_transformer, ["Age", "Fare"]),
        ]
    )
    preprocessed_data = pd.DataFrame(processor.fit_transform(X=df))

    return preprocessed_data