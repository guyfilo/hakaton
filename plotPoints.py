import pandas as pd
from matplotlib import pyplot as plt
import numpy


crimes = pd.read_csv("training set")
values = {"BATTERY":'black', "THEFT":'red', "CRIMINAL DAMAGE":'green' ,"DECEPTIVE PRACTICE":'blue' , "ASSAULT":'pink'}

print(crimes["Primary Type"])

colors = crimes["Primary Type"].apply(lambda x: values[x])
print(colors)

fig, axes = plt.subplots()
plt.scatter(crimes['X Coordinate'][:1000], crimes['Y Coordinate'][:1000], c=colors[:1000])
plt.show()

label = crimes["Primary Type"]
X = crimes.drop(columns="Primary Type")
