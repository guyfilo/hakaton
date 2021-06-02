import pandas as pd
import datetime as dt
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sklearn.model_selection

"""df = pd.read_csv("dataset_crimes.csv")
v = pd.unique(df['Primary Type'])
u = pd.unique(df['Location Description'])
df = df.iloc[:, 1:]"""

primary_types = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
primary_types_names = ["BATTERY", "THEFT", "CRIMINAL DAMAGE", "DECEPTIVE PRACTICE", "ASSAULT"]

"""beat, location = df['Beat'], df['Location']

beat_to_location = {beat[j]: location[j] for j in range(len(beat))}

train = df.sample(frac=0.8, random_state=200)
test_validation = df.drop(train.index)
validation = df.sample(frac=0.5, random_state=200)
test = df.drop(validation.index)

pd.DataFrame.to_csv(train, "training set")
pd.DataFrame.to_csv(validation, "validation set")
pd.DataFrame.to_csv(test, "test set")"""

df = pd.read_csv("training set", index_col=0)

df['Arrest'] = df['Arrest'].astype(int)
df['Domestic'] = df['Domestic'].astype(int)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = df['Date'].dt.hour
df.dropna(inplace=True)
df.replace({'Primary Type': primary_types}, inplace=True)


def hourCrime(hour):
    newDf = df.loc[df['Hour'].astype(int) == hour]
    #print(newDf)
    types = []
    for j in range(5):
        #print(np.array(newDf['Primary Type'] == j))
        types.append(np.count_nonzero(newDf['Primary Type'] == j))
    plt.ylim(0, 700)
    plt.bar(primary_types_names, types)
    plt.title("hour = " + str(hour))
    plt.xticks(range(5))
    plt.show()


values = {0:'black', 1:'red', 2:'green', 3:'blue' , 4:'pink'}
#colors = df["Primary Type"].apply(lambda x: values[x])


def crime_hour_location(hour, crime):
    newDf = df.loc[(df['Hour'].astype(int) == hour) & (df['Primary Type'] == crime)]
    colors = newDf["Primary Type"].apply(lambda x: values[x])
    patches = []
    for j in range(5):
        patches.append(mpatches.Patch(Color=values[j], label=primary_types_names[j]))
    #patch = mpatches.Patch(Color='black', label='Battery')
    plt.scatter(newDf['X Coordinate'], newDf['Y Coordinate'], c=colors, s=[7]*len(newDf))
    plt.legend(handles=patches)
    plt.title("hour = " + str(hour))
    plt.show()

for i in range(5):
    for j in range(24):
        crime_hour_location(j,i)


#correlations = df.corr()
#print(correlations)

