# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
import location_filter
from sklearn.tree import DecisionTreeClassifier

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}


def pre_proccing():
    df = pd.read_csv("training set")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Hour'] = df['Date'].dt.hour
    df = location_filter.create_list(df, df)
    df['Arrest'] = df['Arrest'].astype(int)
    df['Domestic'] = df['Domestic'].astype(int)
    df = pd.get_dummies(df, columns=['Block'], drop_first=True)
    df.drop(columns=['IUCR', 'FBI Code', 'Description', 'Case Number'], inplace=True)
    response_vector = df['Primary Type'].apply(lambda x: values[x]).to_numpy()
    df.drop(columns=['Primary Type'], inplace=True)
    training_point = df.to_numpy()
    return training_point, response_vector


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    X, y = pre_proccing()
    tree = DecisionTreeClassifier(max_depth=5)
    scores = tree.fit(X, y).score(X, y)
    print(scores)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# hosafti comment
