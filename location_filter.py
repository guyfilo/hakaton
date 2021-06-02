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
    k_neighbors = KNeighborsClassifier(n_neighbors=1)
    k_neighbors.fit(locations, response_vector)
    return k_neighbors,locations,response_vector


def create_list(x, y):
    x = change_data(training).predict_proba([[x, y]])
    print(x)


if __name__ == '__main__':
    training = pd.read_csv("training set")
    a ,b,c=change_data(training)

    print(a.score(b,c))
