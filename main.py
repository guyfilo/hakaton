# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.tree as tr
import  sklearn.ensemble as en
from preprocess import *
from sklearn.ensemble import BaggingClassifier
from location_filter import *

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"
#(['X Coordinate', 'Y Coordinate'], 'block', 'Block'),(['X Coordinate', 'Y Coordinate'], 'beat', 'Beat')
prod_feacture_args = [(['X Coordinate', 'Y Coordinate'], 'location', 'Primary Type', 1),
                      (['DayTime'], 'dayTime', 'Primary Type', 15) ]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train = pd.read_csv('training set')
    data_pro = PreProcessing(train)
    X, y = data_pro.load_new_features(data_pro.training_data, True)
    val = pd.read_csv('test set')
    Xv, yv = data_pro.load_new_features(val, False)
    tree = BaggingClassifier(tr.DecisionTreeClassifier(max_depth=20), n_estimators=66)
    tree.fit(X, y)
    print(tree.score(X, y))
    print(tree.score(Xv, yv))
    rand_forest = en.RandomForestClassifier(n_estimators=200)
    rand_forest.fit(X, y)
    print("rnd forest")
    print(rand_forest.score(X, y))
    print(rand_forest.score(Xv, yv))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# hosafti comment
