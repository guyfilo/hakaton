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
training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"
# ,(['X Coordinate', 'Y Coordinate'], 'block', 'Block',1)(['X Coordinate', 'Y Coordinate'], 'beat', 'Beat',1)
prod_feacture_args = [(['X Coordinate', 'Y Coordinate'], 'location', 'Primary Type', 100),
                      (['DayTime'], 'dayTime', 'Primary Type', 100),(['X Coordinate', 'Y Coordinate'], 'beat', 'Beat',4)]



class PreProcessing:

    def __init__(self, training_data):
        self.training_data = training_data
        self.prob_feature = [ProbFeature(*args) for args in prod_feacture_args]
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
        df.drop(columns=['Date', 'Block', 'IUCR', 'FBI Code', 'Description', 'Case Number', 'Location Description', 'Updated On',
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
        data.drop(columns=['X Coordinate', 'Y Coordinate', 'DayTime', 'Beat', 'Primary Type'], inplace=True)
        if not(is_training):
            missing_cols =  set(self.training_data.columns)-set(data.columns)
            # Add a missing column in test set with default value equal to 0
            for c in missing_cols:
                data[c] = 0
            # Ensure the order of column in the test set is in the same order than in train set
            data = data[self.training_data.columns]
        return data, response_vector


if __name__ == '__main__':
    train = pd.read_csv('training set')
    data_pro = PreProcessing(train)
    X, y = data_pro.load_new_features(data_pro.training_data, True)
    val = pd.read_csv('validation set')
    Xv, yv = data_pro.load_new_features(val, False)
    p = en.BaggingClassifier(KNeighborsClassifier(n_neighbors=6),n_estimators=3)
    print("tree")
    p.fit(X, y)
    print(p.score(X, y))
    print(p.score(Xv, yv))
    tree = en.BaggingClassifier(tr.DecisionTreeClassifier(max_depth=20),n_estimators=30)
    tree.fit(X, y)
    print("tree")
    print(tree.score(X, y))
    print(tree.score(Xv, yv))
    rand_forest = en.RandomForestClassifier(n_estimators=200,max_depth=10)
    rand_forest.fit(X,y)
    print("rnd forest")
    print(rand_forest.score(X, y))
    print(rand_forest.score(Xv, yv))





