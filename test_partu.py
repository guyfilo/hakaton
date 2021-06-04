
import numpy.linalg as lin
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def check_result(result, data, date):
    timing = pd.to_datetime(data['Date'], errors='coerce')
    data["date"] = timing.dt.date
    data['time'] = timing
    data = data[data["date"] == date]
    data["visited"] = False
    total_crimes = data.shape[0]
    for k in result:
        data["visited"] = data.apply(check(k), axis=1)
    return np.count_nonzero(data["visited"]) / total_crimes


def check(p1):
    def checkkk(p2):
        if lin.norm([p1[0] - p2["X Coordinate"], p1[1] - p2["Y Coordinate"]]) >= 500:
            return False
        delta = p1[2] - p2["time"]
        if delta.seconds / 3600.0 > 0.5 or p2['visited']:
            return False
        return True
    return checkkk





if __name__ == '__main__':
    X = pd.read_csv("training set")
    X['time'] = pd.to_datetime(X['Date'], errors='coerce')
    x = X.iloc[0]

    result = [(x["X Coordinate"], x["Y Coordinate"], x["time"])]
    print(x["time"].date())
    print(check_result(result, X, x["time"].date()))
