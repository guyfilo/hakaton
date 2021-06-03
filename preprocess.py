import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.tree as tr
import sklearn.ensemble as en
from location_filter import *

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"
#(['X Coordinate', 'Y Coordinate'], 'block', 'Block'),(['X Coordinate', 'Y Coordinate'], 'beat', 'Beat')
prod_feacture_args = [(['X Coordinate', 'Y Coordinate'], 'location', 'Primary Type', 10),
                      (['DayTime'], 'dayTime', 'Primary Type', 30) ]



class PreProcessing:

    def __init__(self, training_data):
        self.training_data = training_data
        self.prob_feature = [ProbFeature(*args) for args in prod_feacture_args]
        print(type(self.prob_feature[0]))
        self.training_data =  self.basic_preprocessing(self.training_data)


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
        df.drop(columns=['Date', 'IUCR', 'FBI Code', 'Description', 'Case Number', 'Location Description', 'Updated On',
                         'Location'], inplace=True)

        return df

    def load_new_features(self, data, is_training):
        if is_training:
            for prob in self.prob_feature:
                prob.fit(data)
        else:
            data = self.basic_preprocessing(data)
        for prob in self.prob_feature:
            prob.add_features(data)
        response_vector = data['Primary Type'].apply(lambda x: values[x])
        data.drop(columns=['Block', 'X Coordinate', 'Y Coordinate', 'DayTime', 'Beat', 'Primary Type'], inplace=True)
        return data, response_vector


if __name__ == '__main__':
    train = pd.read_csv('training set')
    data_pro = PreProcessing(train)
    X, y = data_pro.load_new_features(data_pro.training_data, True)
    val = pd.read_csv('validation set')
    Xv, yv = data_pro.load_new_features(val, False)
    tree = tr.DecisionTreeClassifier(max_depth=10)
    tree.fit(X, y)
    print("tree")
    print(tree.score(X, y))
    print(tree.score(Xv, yv))
    rand_forest = en.RandomForestClassifier(n_estimators=200)
    rand_forest.fit(X,y)
    print("rnd forest")
    print(rand_forest.score(X, y))
    print(rand_forest.score(Xv, yv))





