import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
training_set_locations =[]
values = {"BATTERY":0, "THEFT":1, "CRIMINAL DAMAGE":2  ,"’DECEPTIVE PRACTICE":3 , "ASSAULT":4}
def change_data(data_frame):

