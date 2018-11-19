from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from data_preprocessing import x_list, y
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
import pandas as pd


x_rs = x_list[4][1]
x_train_rs, x_test_rs,y_train, y_test = train_test_split(x_rs,y,test_size=0.33, random_state=0)


class MartinRegression(object):

    algorithms = [SVR(C=1.4, kernel='linear', max_iter=14, epsilon=0.132), Ridge(alpha=0.4, fit_intercept=True, max_iter=35, solver='sag'), LinearRegression()]


    def __init__(self):
        pass

    def fit(cls,x,y):
        for i, algorithm in enumerate(cls.algorithms):
            algorithm.fit(x,y)

    def predict(cls,x):
        predictions = np.arange(len(x)*3).reshape(len(x),3)
        predictions = predictions.astype(float)
        final_predictions = []

        for i, algorithm in enumerate(cls.algorithms):
            predictions[:,i] = algorithm.predict(x)

        for i in range(len(predictions)):
            final_predictions.append(predictions[i].mean())

        return final_predictions


if __name__ == '__main__':
    mr = MartinRegression()
    mr.fit(x_train_rs,y_train)
    y_pred = mr.predict(x_test_rs)

    print(f'MR Result: {mean_squared_error(y_test, y_pred)}\n')
