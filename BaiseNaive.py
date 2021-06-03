import pandas as pd
import re
import numpy as np
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}

class BayesNaive:

    def __init__(self, data, param_name, split=True):
        self.big_bag = pd.Series([y for x in data[param_name].values.astype(str).flatten() for y in
                                  re.split('[^A-Za-z]', x)]).value_counts().drop([''])
        self.param_name = param_name
        d = {word: np.zeros(5) for word in self.big_bag.keys()}
        for j in range(5):
            data_prime = data[data['Primary Type'] == j]
            local_bag = pd.Series([y for x in data_prime['Location Description'].values.astype(str).flatten() for y in
                                   re.split('[^A-Za-z]', x)]).value_counts().drop([''])
            for key in local_bag.keys():
                d[key][j] = local_bag[key] / self.big_bag[key]
        self.d = pd.DataFrame(data=d)
        self.add_features(data)


    def predict(self, x):
        words = re.split("[^A-Za-z]", x)
        words = [self.d[word] if word in self.d.keys() else 0.2 * np.ones(5) for word in words]
        return pd.Series(np.average(words, axis=0))

    """maxes = {word: np.zeros(2) for word in words}
    local_df = pd.DataFrame(maxes)
    for word in words:
        if word in self.d.keys():
            local_df[word][0] = np.argmax(self.d[word])
            local_df[word][1] = self.d[maxes][maxes[0]]
    if np.count_nonzero(local_df[:, 1] > 0) > 0:
        return local_df.idxmax(axis=1)[1]
    return np.random.randint(0, 4)"""

    def add_features(self, data):
        data[[f"{self.param_name} 0", f"{self.param_name} 1",
              f"{self.param_name} 2", f"{self.param_name} 3",
              f"{self.param_name} 4"]] = np.array(data[self.param_name].apply(self.predict))


    def accuracy(self, X, y):
        y_hat = X.apply(self.predict)
        return np.count_nonzero(y == y_hat) / len(y)

if __name__ == '__main__':
    print("lpadlp")
    X = pd.read_csv("training set", index_col=0)
    X['Primary Type'] = X['Primary Type'].apply(lambda x: values[x])
    X.dropna(inplace=True)
    b = BayesNaive(X, 'Location Description')
    print(X)


