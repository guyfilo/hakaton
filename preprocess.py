import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.tree as tr
from location_filter import *
import sklearn.ensemble as en
import sklearn.tree as tr
from sklearn.neighbors import KNeighborsClassifier
from location_filter import *
from sklearn.linear_model import LogisticRegression
from BaiseNaive import *

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"
# ,(['X Coordinate', 'Y Coordinate'], 'block', 'Block',1)(['X Coordinate', 'Y Coordinate'], 'beat', 'Beat',1)
prod_feacture_args = [(['X Coordinate', 'Y Coordinate'], 'location', 'Primary Type', 100),
                      (['DayTime'], 'dayTime', 'Primary Type', 100)]



class PreProcessing:

    def __init__(self, training_data):
        self.training_data = training_data
        self.prob_feature = [ProbFeature(*args) for args in prod_feacture_args]
        self.training_data = self.basic_preprocessing(self.training_data)
        self.ld_prob = BayesNaive(self.training_data, 'Location Description')


    def basic_preprocessing(self, df):
        df.dropna(inplace=True)
        time = pd.to_datetime(df['Date'], errors='coerce')
        df['DayTime'] = time.dt.hour + (time.dt.minute / 60)
        df['WeekDay'] = time.dt.weekday
        df['Month'] = time.dt.month
        df['MonthDay'] = time.dt.day
        df['Arrest'] = df['Arrest'].astype(int)
        df['Domestic'] = df['Domestic'].astype(int)
        df = pd.get_dummies(df, columns=[ 'WeekDay', 'Month', 'MonthDay'], drop_first=True)
        df.drop(columns=['Date', 'Block', 'IUCR', 'FBI Code', 'Description', 'Case Number', 'Updated On',
                         'Location'], inplace=True)
        df['Primary Type'] = df['Primary Type'].apply(lambda x: values[x])
        return df

    def load_new_features(self, data, is_training):
        if is_training:
            for prob in self.prob_feature:
                prob.fit(data)
        else:
            data = self.basic_preprocessing(data)
            self.ld_prob.add_features(data)
        for prob in self.prob_feature:
            prob.add_features(data)
        response_vector = data['Primary Type']
        data.drop(columns=['X Coordinate', 'Y Coordinate', 'DayTime', 'Beat', 'Primary Type', 'Location Description'],
                  inplace=True)
        if not is_training:
            missing_cols = set(self.training_data.columns)-set(data.columns)
            # Add a missing column in test set with default value equal to 0
            for c in missing_cols:
                data[c] = 0
            # Ensure the order of column in the test set is in the same order than in train set
            data = data[self.training_data.columns]
        return data, response_vector


if __name__ == '__main__':
    train = pd.read_csv('training set', index_col=0)
    data_pro = PreProcessing(train)
    X, y = data_pro.load_new_features(data_pro.training_data, True)
    val = pd.read_csv('validation set', index_col=0)
    Xv, yv = data_pro.load_new_features(val, False)






