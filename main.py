# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
training_set_locations =[]

def pre_proccing(name):
    # Use a breakpoint in the code line below to debug your script.
    training = pd.read_csv("test.csv")
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['date'] = ((data['date'] - datetime(1970, 1, 1)).dt.total_seconds()) / (24 * 60 * 60)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pre_proccing("a")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
#hosafti comment