import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}


class BlockRate:

    def __init__(self, data, param_name='Block', split=True):
        D = ['S', 'N', 'E', 'W']
        self.big_bag = pd.Series(data[param_name].apply(lambda x: x[6])).value_counts()
        self.param_name = param_name
        d = {word: np.zeros(5) for word in self.big_bag.keys()}
        for j in range(5):
            data_prime = data[data['Primary Type'] == j]
            local_bag = pd.Series(data_prime[param_name].apply(lambda x: x[6])).value_counts()
            for key in local_bag.keys():
                d[key][j] = local_bag[key] / self.big_bag[key]
        self.d = pd.DataFrame(data=d)
        self.add_features(data)

    def add_features(self, data):
        data[f"{self.param_name} 0"] = False
        data[f"{self.param_name} 1"] = False
        data[f"{self.param_name} 2"] = False
        data[f"{self.param_name} 3"] = False
        data[f"{self.param_name} 4"] = False
        data[[f"{self.param_name} 0", f"{self.param_name} 1",
              f"{self.param_name} 2", f"{self.param_name} 3",
              f"{self.param_name} 4"]] = np.array(data[self.param_name].apply(
            lambda x : self.d[x[6]]
        ))


if __name__ == '__main__':
    X = pd.read_csv("training set")
    X['Primary Type'] = X['Primary Type'].apply(lambda x: values[x])
    kkk = BlockRate(X)
    kkk.add_features(X)
    print(X)