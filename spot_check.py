from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from data_preprocessing import x_list, y
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import warnings
import numpy as np


x = x_list[0][1]
x_rs_cd = x_list[5][1]

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0)
x_train_rs, x_test_rs,y_train,y_test = train_test_split(x_rs_cd,y,test_size=0.33, random_state=0)


def test_harness(model_list,x_list,y):
    for name, model in model_list:
        model = model()
        for tname, x in x_list:
            scoring = 'neg_mean_squared_error'
            kfold = KFold(n_splits=10, random_state=7)
            result = cross_val_score(model,x,y,cv=kfold,scoring=scoring)
            print(f'{name} with {tname} result: {result.mean()}')
        print('\n')

def test_harness_tuning(x,y,model,param_grid):
    best_params = []
    model = model()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=0.1).fit(x,y)
    best_params.append(grid.best_params_)
    print(best_params)

def model_evulation(model_list,x_list,y, y_t):
    for name, model in model_list:
        for x_train, x_test in x_list:
            model.fit(x_train,y)
            y_pred = model.predict(x_test)
            print(f'{name} MSE score: {mean_squared_error(y_t, y_pred)}')
        print('\n')


with warnings.catch_warnings():
    warnings.simplefilter('ignore')



    param_grid = dict()

    model_list = []
    model_list.append(('SVR', SVR()))
    model_list.append(('Ridge', Ridge()))
    model_list.append(('RFR', RandomForestRegressor()))
    model_list.append(('LR', LinearRegression()))

    x_list = [(x_train, x_test), (x_train, x_test), (x_train_rs, x_test_rs), (x_train,x_test)]

    model_evulation(model_list,x_list,y_train, y_test)
