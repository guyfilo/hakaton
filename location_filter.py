import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
training_set_locations =[]
values = {"BATTERY":0, "THEFT":1, "CRIMINAL DAMAGE":2  ,"’DECEPTIVE PRACTICE":3 , "ASSAULT":4}
def change_data(data_frame):
    response_vector = data_frame['Primary Type'].to_numpy().apply(lambda x: values[x])
    loaction_x = data_frame['X Coordinate'].to_numpy()
    loaction_y = data_frame['T Coordinate'].to_numpy()
    locations = np.append(np.array([loaction_y]).T, np.array([loaction_x]).T, 1)
    k_neighbors = KNeighborsClassifier(n_neighbors=10)
    k_neighbors.fit(locations,response_vector)
    return  k_neighbors
def create_list(x,y):
    d,i = change_data(training)



if __name__ == '__main__':
    training = pd.read_csv("test.csv")
    change_data(training)







