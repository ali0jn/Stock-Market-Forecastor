import math, datetime, pickle
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style


class Predictor:
    def __init__(self):
        self.main()

    def main(self):
        print(' ---- Welcome to the Stock Market Prediction Tool ---- ')
        loadData = input('\n\nEnter 1 in the console to load your training data, make sure you saved your data under name data.csv in your current working directory: ')
        if loadData == '1':
            self.dataFrame = DataFrame.from_csv('data.csv')
            self.dataFrame = self.dataFrame[['Open', 'High', 'Low', 'Adj Close', 'Volume', ]]
            self.dataFrame['HL_%'] = (self.dataFrame['High'] - self.dataFrame['Adj Close']) / self.dataFrame[
                'Adj Close'] * 100.0
            self.dataFrame['PCT_change'] = (self.dataFrame['Adj Close'] - self.dataFrame['Open']) / self.dataFrame[
                'Open'] * 100.0
            self.dataFrame = self.dataFrame[['Adj Close', 'HL_%', 'PCT_change', 'Volume']]
        dataSize = input('\n\n\nWhat volume of data you want to be considered, write it in two decimal points. i.e for 25% of data you can write 0.25: ')
        self.predict(float(dataSize))

    def predict(self, data_size):
        self.forecastColumn = 'Adj Close'
        self.dataFrame.fillna(-99999, inplace=True)
        self.forecastOut = int(math.ceil(0.01 * len(self.dataFrame)))
        self.dataFrame['Label'] = self.dataFrame[self.forecastColumn].shift(-self.forecastOut)
        self.features = np.array(self.dataFrame.drop('Label', 1))
        self.features = preprocessing.scale(self.features)
        self.features = self.features[:-self.forecastOut]
        self.predictedFeatures = self.features[-self.forecastOut:]
        self.dataFrame.dropna(inplace=True)
        self.labels = np.array(self.dataFrame['Label'])
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(self.features, self.labels, test_size=data_size)
        self.classifier = LinearRegression(n_jobs=-1)
        self.classifier.fit(self.X_train, self.y_train)
        self.storeClassifier(self.classifier)
        pickle_in = open('linearregression.pickle', 'rb')
        self.classifier = pickle.load(pickle_in)
        self.accuracy = self.classifier.score(self.X_test, self.y_test)
        self.predictSet = self.classifier.predict(self.predictedFeatures)
        self.dataFrame['Prediction'] = np.nan
        self.setTimeline()
        self.plotData()

    def storeClassifier(self, classifier):
        with open('linearregression.pickle', 'wb') as file:
            pickle.dump(classifier, file)

    def setTimeline(self):
        last_date = self.dataFrame.iloc[-1].name
        last_unix = last_date.timestamp()
        one_day = 86400
        next_unix = last_unix + one_day

        for i in self.predictSet:
            next_date = datetime.datetime.fromtimestamp(next_unix)
            next_unix += one_day
            self.dataFrame.loc[next_date] = [np.nan for _ in range(len(self.dataFrame.columns) - 1)] + [i]

    def plotData(self):
        style.use('ggplot')
        self.dataFrame['Adj Close'].plot()
        self.dataFrame['Prediction'].plot()
        plt.legend(loc=4)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()


obj = Predictor()
