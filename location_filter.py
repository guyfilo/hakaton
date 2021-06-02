import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}

def change_data(data_frame):
    data_frame.dropna(inplace=True)
    response_vector = data_frame['Primary Type'].apply(lambda x: values[x]).to_numpy()
    loaction_x = data_frame['X Coordinate'].to_numpy()
    loaction_y = data_frame['Y Coordinate'].to_numpy()
    locations = np.append(np.array([loaction_y]).T, np.array([loaction_x]).T, 1)
    k_neighbors = KNeighborsClassifier(n_neighbors=20)
    k_neighbors.fit(locations, response_vector)
    return k_neighbors, locations, response_vector


def create_list(data, training):
    x = change_data(training).predict_proba(data)
    data['Propability_location_based_B'] = x[:, 0]
    data['Propability_location_based_T'] = x[:, 1]
    data['Propability_location_based_C'] = x[:, 2]
    data['Propability_location_based_D'] = x[:, 3]
    data['Propability_location_based_A'] = x[:, 4]
    return data


if __name__ == '__main__':
    k, l, r = change_data(training)
    print(k.score(l, r))
