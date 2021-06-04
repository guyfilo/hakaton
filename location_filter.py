import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

training_set_locations = []
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}

from math import radians, cos, sin, asin, sqrt


def haversine(a, b):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = a[0], a[1], b[0], b[1]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


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
        if (self.column_title == "log"):
            k_neighbors = KNeighborsClassifier(n_neighbors=10, metric=lambda a, b: haversine(a, b))
        else:
            k_neighbors = KNeighborsClassifier(n_neighbors=self.k, algorithm='kd_tree')
        k_neighbors.fit(X, response_vector)
        self.location_model = k_neighbors

    def add_features(self, test_data):
        prob_predict = self.location_model.predict_proba(test_data[self.features])
        for label, index in self.label_map.items():
            test_data[f'Propability_{self.column_title}_based_{label}'] = prob_predict[:, index]


if __name__ == '__main__':
    a = haversine([-87.642811511,41.706219652],[-87.703428872,41.963590636])
    print(a)