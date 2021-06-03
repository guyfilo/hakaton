import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection

df = pd.read_csv("dataset_crimes.csv")
v = pd.unique(df['Primary Type'])
u = pd.unique(df['Location Description'])

primary_types = {j: v[j] for j in range(5)}

beat, location = df['Beat'], df['Location']

beat_to_location = {beat[j]: location[j] for j in range(len(beat))}

train = df.sample(frac=0.8, random_state=200)
test_validation = df.drop(train.index)
validation = df.sample(frac=0.5, random_state=200)
test = df.drop(validation.index)

pd.DataFrame.to_csv(train, "training set")
pd.DataFrame.to_csv(validation, "validation set")
pd.DataFrame.to_csv(test, "test set")

from location_filter import *

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"

prod_feacture_args = [(['X Coordinate', 'Y coordinate'], 'location', 'Primary Type'),
                      (['X Coordinate', 'Y coordinate'], 'block', 'Block'),
                      (['DayTime'], 'dayTime', 'Primary Type'),
                      (['X Coordinate', 'Y coordinate'], 'beat', 'Beat')]



class PreProcessing:

    def __init__(self, training_data):
        self.training_data = training_data
        self.prob_feature = [ProbFeature()]
        self.basic_preprocessing(self.training_data)

    def basic_preprocessing(self, df):
        df.dropna(inplace=True)
        time = pd.to_datetime(df['Date'], errors='coerce')
        df['DayTime'] = time.dt.hour + (time.dt.minute / 60)
        df['WeekDay'] = time.dt.weekday
        df['Month'] = time.dt.month
        df['MonthDay'] = time.dt.day
        df['Arrest'] = df['Arrest'].astype(int)
        df['Domestic'] = df['Domestic'].astype(int)
        df = pd.get_dummies(df, columns=['Block', 'WeekDay', 'Month', 'MonthDay'], drop_first=True)
        df.drop(columns=['Date', 'IUCR', 'FBI Code', 'Description', 'Case Number', 'Location Description', 'Updated On', 'Location', 'Block'], inplace=True)
        response_vector = df['Primary Type'].apply(lambda x: values[x])
        return df

    def load_new_features(self, training):