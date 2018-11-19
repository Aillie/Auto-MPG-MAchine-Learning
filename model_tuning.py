from sklearn.svm import SVR
from data_preprocessing import x_list,y
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
import numpy as np
import warnings


x = x_list[0][1]
x_rs_cd = x_list[5][1]
x_std = x_list[2][1]
x_cd_std = x_list[3][1]
x_rs = x_list[4][1]


def tuning(x,y,model,param_grid):
    best_params = []
    model = model()
    scoring = 'neg_mean_squared_error'
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=10, n_jobs=3, return_train_score=False)
    grid.fit(x,y)
    best_params.append(grid.best_params_)
    print(best_params)

def model_evaluation(x,y,model):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_pred_t = model.predict(x_train)
    result = mean_squared_error(y_test, y_pred)
    result_t = mean_squared_error(y_train, y_pred_t)
    print('Test Result:', result)
    print('\nTrain Result:', result_t)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')


        model_evaluation(x,y,LinearRegression())
