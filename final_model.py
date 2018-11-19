from data_preprocessing import x_list, y
from ensembles_tuning import MartinRegression
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_squared_error
import warnings


x_rs = x_list[4][1]
x_train,x_test, y_train, y_test = train_test_split(x_rs,y,test_size=0.33, random_state=0)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    
    mr = MartinRegression()
    mr.fit(x_train, y_train)

    filename = 'MartinRegressor.sav'
    pickle.dump(mr,open(filename, 'wb'))

    model = pickle.load(open(filename, 'rb'))
    y_pred = model.predict(x_test)
    print('MartinRegressor result: ', mean_squared_error(y_test, y_pred))
