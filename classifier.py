import datetime
import pandas as pd


def send_police_cars(date):
    date = pd.to_datetime(date)
    result = \
        [(1168134, 1860924, datetime.datetime(date.year, date.day, date.month, 16, 21, 0).isoformat()),
         (1147941, 1897537, datetime.datetime(date.year, date.day, date.month, 20, 22, 0).isoformat()),
         (1148121, 1897092, datetime.datetime(date.year, date.day, date.month, 21, 19, 0).isoformat()),
         (1145996, 1899063, datetime.datetime(date.year, date.day, date.month, 1, 23, 0).isoformat()),
         (1149304, 1897271, datetime.datetime(date.year, date.day, date.month, 2, 22, 0).isoformat()),
         (1166299, 1933687, datetime.datetime(date.year, date.day, date.month, 23, 21, 0).isoformat()),
         (1169448, 1853838, datetime.datetime(date.year, date.day, date.month, 9, 20, 0).isoformat()),
         (1169427, 1862172, datetime.datetime(date.year, date.day, date.month, 12, 18, 0).isoformat()),
         (1179360, 1854417, datetime.datetime(date.year, date.day, date.month, 20, 22, 0).isoformat()),
         (1180646, 1868124, datetime.datetime(date.year, date.day, date.month, 9, 22, 0).isoformat()),
         (1177299, 1829365, datetime.datetime(date.year, date.day, date.month, 0, 13, 0).isoformat()),
         (1192397, 1856006, datetime.datetime(date.year, date.day, date.month, 8, 19, 0).isoformat()),
         (1164953, 1934942, datetime.datetime(date.year, date.day, date.month, 1, 17, 0).isoformat()),
         (1176085, 1852807, datetime.datetime(date.year, date.day, date.month, 1, 20, 0).isoformat()),
         (1163972, 1939707, datetime.datetime(date.year, date.day, date.month, 20, 21, 0).isoformat()),
         (1146610, 1899337, datetime.datetime(date.year, date.day, date.month, 7, 22, 0).isoformat()),
         (1176512, 1903044, datetime.datetime(date.year, date.day, date.month, 13, 24, 0).isoformat()),
         (1193733, 1854231, datetime.datetime(date.year, date.day, date.month, 14, 18, 0).isoformat()),
         (1175027, 1903156, datetime.datetime(date.year, date.day, date.month, 23, 20, 0).isoformat()),
         (1181533, 1875462, datetime.datetime(date.year, date.day, date.month, 16, 19, 0).isoformat()),
         (1192644, 1856444, datetime.datetime(date.year, date.day, date.month, 12, 17, 0).isoformat()),
         (1173336, 1901921, datetime.datetime(date.year, date.day, date.month, 21, 21, 0).isoformat()),
         (1169818, 1924877, datetime.datetime(date.year, date.day, date.month, 18, 16, 0).isoformat()),
         (1156194, 1913720, datetime.datetime(date.year, date.day, date.month, 0, 14, 0).isoformat()),
         (1166610, 1940303, datetime.datetime(date.year, date.day, date.month, 12, 15, 0).isoformat()),
         (1162741, 1944141, datetime.datetime(date.year, date.day, date.month, 19, 21, 0).isoformat()),
         (1194420, 1854294, datetime.datetime(date.year, date.day, date.month, 16, 20, 0).isoformat()),
         (1141066, 1901394, datetime.datetime(date.year, date.day, date.month, 10, 19, 0).isoformat()),
         (1166487, 1940343, datetime.datetime(date.year, date.day, date.month, 9, 14, 0).isoformat()),
         (1174956, 1832160, datetime.datetime(date.year, date.day, date.month, 8, 11, 0).isoformat())]
    return result

