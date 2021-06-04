import pandas as pd
import numpy as np

df = pd.read_csv("dataset_crimes.csv", index_col=0)
df_2 = pd.read_csv("crimes_dataset_part2.csv", index_col=0)
df = pd.concat([df, df_2])
df.to_csv("data")


train = df.sample(frac=0.8, random_state=88)
test = df.drop(train.index)


pd.DataFrame.to_csv(train, "training set")
pd.DataFrame.to_csv(test, "test set")
