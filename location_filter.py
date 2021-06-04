import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}


def change_data(training):
    training.dropna(inplace=True)
    response_vector = training['Primary Type'].apply(lambda x: values[x]).to_numpy()
    loaction_x = training['X Coordinate'].to_numpy()
    loaction_y = training['Y Coordinate'].to_numpy()
    locations = np.append(np.array([loaction_y]).T, np.array([loaction_x]).T, 1)
    k_neighbors = KNeighborsClassifier(n_neighbors=20)
    k_neighbors = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
    k_neighbors.fit(locations, response_vector)
    return k_neighbors,locations

def nearest_time(data_frame):
    data_frame.dropna(inplace=True)
    response_vector = data_frame['Primary Type'].apply(lambda x: values[x]).to_numpy()
    time = pd.to_datetime(data_frame['Date'], errors='coerce')
    print(time)
    time = time.dt.hour + (time.dt.minute / 60)
    print(np.array([time]).T)
    time = np.array([time]).T
    k_neighbors = KNeighborsClassifier(n_neighbors=20, algorithm='ball_tree')
    k_neighbors.fit(time, response_vector)
    return k_neighbors, time, response_vector

def create_list(data, training):
    l, location = change_data(training)
    x = l.predict_proba(location)
    data['Propability_location_based_B'] = x[:, 0]
    data['Propability_location_based_T'] = x[:, 1]
    data['Propability_location_based_C'] = x[:, 2]
    data['Propability_location_based_D'] = x[:, 3]
    data['Propability_location_based_A'] = x[:, 4]
    return data


class ProbFeature:

    def __init__(self, features, column_title, feature_label='Primary Type', k=1):
        self.type = None
        self.k = k
        self.location_model = None
        self.label_map = None
        self.features = features
        self.feature_label = feature_label
        self.column_title = column_title

    def fit(self, training_data):
        training_data.dropna(inplace=True)
        self.label_map = {w: i for i, w in enumerate(training_data[self.feature_label].unique())}
        response_vector = training_data[self.feature_label].apply(lambda x: self.label_map[x])
        X = training_data[self.features]
        k_neighbors = KNeighborsClassifier(n_neighbors=self.k, algorithm='ball_tree', radius=500)
        k_neighbors.fit(X, response_vector)
        self.location_model = k_neighbors

    def add_features(self, test_data):
        prob_predict = self.location_model.predict_proba(test_data[self.features])
        for label, index in self.label_map.items():
            test_data[f'Propability_{self.column_title}_based_{label}'] = prob_predict[:, index]


if __name__ == '__main__':
    df_train = pd.read_csv('training set')
    df_val = pd.read_csv('validation set')
    df_train.dropna(inplace=True)
    df_val.dropna(inplace=True)
    location_knn = ProbFeature(['X Coordinate', 'Y Coordinate'], 'location')
    location_knn.fit(df_train)
    location_knn.add_features(df_val)









