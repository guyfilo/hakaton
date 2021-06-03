import pandas as pd
import numpy as np

df = pd.read_csv("dataset_crimes.csv", index_col=0)
df_2 = pd.read_csv("crimes_dataset_part2.csv", index_col=0)
df = pd.concat([df, df_2])


train = df.sample(frac=0.7, random_state=200)
test_valid = df.drop(train.index)
validation = test_valid.sample(frac=0.5, random_state=200)
test = test_valid.drop(validation.index)


pd.DataFrame.to_csv(train, "training set")
pd.DataFrame.to_csv(validation, "validation set")
pd.DataFrame.to_csv(test, "test set")




"""
v = pd.unique(df['Primary Type'])
u = pd.unique(df['Location Description'])

primary_types = {j: v[j] for j in range(5)}

beat, location = df['Beat'], df['Location']

beat_to_location = {beat[j]: location[j] for j in range(len(beat))}
"""

"""train = df.sample(frac=0.8, random_state=200)
test_validation = df.drop(train.index)
validation = df.sample(frac=0.5, random_state=200)
test = df.drop(validation.index)

pd.DataFrame.to_csv(train, "training set")
pd.DataFrame.to_csv(validation, "validation set")
pd.DataFrame.to_csv(test, "test set")"""