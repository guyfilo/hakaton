import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import haversine_distances
from math import radians
from sklearn.cluster import KMeans
import datetime

colors_array = [v for v in colors.cnames.keys()]

df = pd.read_csv("training set")
df.dropna(inplace=True)
df.drop(columns=['IUCR', 'FBI Code', 'Location', 'Ward', 'Primary Type', 'Block', 'Year', 'FBI Code'], inplace=True)
df['Latitude'] = df['Latitude'].apply(radians)
df['Longitude'] = df['Longitude'].apply(radians)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Hour'] = df['Date'].dt.hour
df['Minute'] = df['Date'].dt.minute
df['WeekDay'] = df['Date'].dt.weekday
df['Cont Hour'] = df['Hour'] + (df['Minute'] / 60)
dfdf = {"hour":[],"amount_inside":[],"amount_in_hour":[], "X_center":[], "Y_center":[], "time_center":[]}
df_xy_time = df[['Latitude', 'Longitude', 'X Coordinate', 'Y Coordinate', 'Cont Hour', 'WeekDay', 'Hour', 'Minute']]

clusters = pd.DataFrame(dfdf)

#for day in range(7):
for hour in range(24):
    #df_jday = df_xy_time[df_xy_time['WeekDay'] == day]
    df_jday = df_xy_time[df_xy_time['Hour'] == hour]
    #df_jday = df_jday[df_jday['Hour'] == hour]
    xy = df_jday[['X Coordinate', 'Y Coordinate']]
    df_jday = df_jday[['Latitude', 'Longitude']]
    db = OPTICS(metric='haversine', max_eps=0.1524, min_cluster_size=50, algorithm='ball_tree').fit(df_jday.to_numpy())
    labels = db.labels_
    fig = plt.figure()
    plt.scatter(xy["X Coordinate"], xy["Y Coordinate"], c=[colors_array[i+10] for i in labels], s=[2]*len(xy))
    plt.title("day " + 'huh' + " hour " + str(hour) + " OPTICS")

    curr = df_xy_time[df_xy_time['Hour'] == hour]
    for label in pd.unique(labels):
        if label != -1:
            clus = curr[labels == label]
            kmeans = KMeans(n_clusters=1, random_state=0).fit(clus[['X Coordinate', 'Y Coordinate']])
            plt.scatter(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], c='black', s=[7])
            d = {"hour": hour, "amount_inside": len(clus),"amount_in_hour": len(curr),
                 "X_center": kmeans.cluster_centers_[0][0], "Y_center": kmeans.cluster_centers_[0][1],
                 "time_center": clus['Minute'].astype(int).mean()}
            clusters = clusters.append(d, ignore_index=True)
    plt.show()

clusters = clusters.sort_values(by="amount_inside", ascending=False)
first_30 = clusters[:30]

first_30['Time'] = first_30.apply( lambda x : datetime.time(int(x["hour"]), int(x["time_center"])), axis=1)
first_30 = first_30[["X_center", "Y_center", "Time"]]

print("h")
