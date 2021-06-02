# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
import location_filter

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
    df.drop(columns=['id', 'price', 'long', 'lat'], inplace=True)
    response_vector = df['Primary Type'].apply(lambda x: values[x]).to_numpy()
    training_point = df.to_numpy()
    return training_point, response_vector


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pre_proccing("a")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# hosafti comment
