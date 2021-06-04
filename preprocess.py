import sklearn.ensemble as en
from location_filter import *
from BaiseNaive import *
from BlockRate import *
import pickle


K_LOCATIONS = 50
K_TIMES = 100
values = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2, "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
" features, column_title, feature_label='Primary Type', k=20"
prod_feacture_args = [(['X Coordinate', 'Y Coordinate'], 'location', 'Primary Type', K_LOCATIONS),
                      (['DayTime'], 'dayTime', 'Primary Type', K_TIMES)]


class PreProcessing:

    def __init__(self, training_data):
        self.training_data = training_data
        self.prob_feature = [ProbFeature(*args) for args in prod_feacture_args]
        self.training_data = self.basic_pre_processing(self.training_data)
        self.ld_prob = BayesNaive(self.training_data, 'Location Description')
        self.blockRate = BlockRate(training_data)

    def basic_pre_processing(self, df, istraining=True):
        if istraining:
            df['Primary Type'] = df['Primary Type'].apply(lambda x: values[x])
        df.dropna(inplace=True)
        time = pd.to_datetime(df['Date'], errors='coerce')
        df['DayTime'] = time.dt.hour + (time.dt.minute / 60)
        df['WeekDay'] = time.dt.weekday
        df['Month'] = time.dt.month
        df['MonthDay'] = time.dt.day
        df['Minute'] = time.dt.minute
        df['Arrest'] = df['Arrest'].astype(int)
        df['Domestic'] = df['Domestic'].astype(int)
        df['Block_val'] = df['Block'].apply(self.calac_by_bloc)
        df.drop(columns=['Date', 'IUCR', 'FBI Code', 'Description', 'Case Number', 'Updated On',
                         'Location', 'ID', 'Ward'], inplace=True)
        return df

    def load_new_features(self, data, is_training):
        if is_training:
            response_vector = data['Primary Type']
            for prob in self.prob_feature:
                prob.fit(data)
        else:
            data = self.basic_pre_processing(data, False)
            self.ld_prob.add_features(data)
            self.blockRate.add_features(data)
        for prob in self.prob_feature:
            prob.add_features(data)
        data.drop(
            columns=['Block', 'X Coordinate', 'Y Coordinate', 'DayTime', 'Beat', 'Location Description',
                     'Longitude', 'Latitude'],
            inplace=True)
        if not is_training:
            missing_cols = set(self.training_data.columns) - set(data.columns)
            # Add a missing column in test set with default value equal to 0
            for c in missing_cols:
                data[c] = 0
            # Ensure the order of column in the test set is in the same order than in train set
            data = data[self.training_data.columns]
        if is_training:
            data.drop(columns=['Primary Type'], inplace=True)
            return data, response_vector
        return data

    def calac_by_bloc(self, block):
        n = int(block[:3])
        d = 1 if block[6] in {'S', 'W'} else -1
        return d * n


if __name__ == '__main__':
    train = pd.read_csv('data', index_col=0)
    data_pro = PreProcessing(train)
    X, y = data_pro.load_new_features(data_pro.training_data, True)
    tree = en.RandomForestClassifier(n_estimators=200, max_depth=20)
    tree.fit(X, y)
    file_1 = open("pickle tree","wb")
    pickle.dump(tree,file_1)
    file_2 = open("pickle Kn", "wb")
    pickle.dump(data_pro, file_2)
