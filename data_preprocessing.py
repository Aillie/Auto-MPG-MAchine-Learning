from data import car_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler, normalize
import matplotlib.pyplot as plt
import warnings


def car_name_fix(data):
    data['company'] = data['car name'].str.split(' ').str[0]
    data = data.drop(['car name'], axis=1)
    return data

def group_company(data):
    data['company'] = data['company'].replace(['chevy','chevroelt'], 'chevrolet')
    data['company'] = data['company'].replace('toyouta', 'toyota')
    data['company'] = data['company'].replace(['vw','vokswagen'], 'volkswagen')
    data['company'] = data['company'].replace('maxda', 'mazda')
    data['company'] = data['company'].replace('mercedes', 'mercedes-benz')
    return data

def hp_fix(data):
    data = data[data['horsepower'] != '?']
    data['horsepower'] = [float(value) for value in data['horsepower']]
    return data

def num_cat(data, feature='origin'):
    for i in range(len(data[feature])):
        data[feature].replace(1, 'America', inplace=True)
        data[feature].replace(2, 'Europa', inplace=True)
        data[feature].replace(3, 'Asia', inplace=True)
    return data

def binarize_discret(data, feature_name):
    feature = data[feature_name]
    feature = pd.get_dummies(feature, prefix=feature_name)
    data = data.drop([feature_name], axis=1)
    data = pd.concat([data,feature], axis=1)
    return data

def cylinder_displacement(data):
    cylinders, displacement = data['cylinders'], data['displacement']
    cylinder_displacement = displacement/cylinders
    data = data.drop(['cylinders', 'displacement'], axis=1)
    data['cylinder_displacement'] = cylinder_displacement
    return data

def rescale(data):
    rs = MinMaxScaler().fit(data)
    data_rs = rs.transform(data)
    return data_rs

def standardize(data):
    sc = StandardScaler().fit(data)
    data_std = sc.transform(data)
    return data_std

def cat_extractor(data):
    cat_data = data.select_dtypes(include=['object'])
    data = data.select_dtypes(exclude=['object'])
    return data, cat_data

def genaralize(data, feature_name):
    feature = data[feature_name]
    value_counts = feature.value_counts()
    to_remove = value_counts[value_counts <= 2].index
    feature.replace(to_remove, 'generalized', inplace=True)
    data[feature_name] = feature
    return data

def find_skew(data):
    print(data.skew())

def log_skew(data):
    for col in data:
        if abs(data[col].skew()) > 0.4:
            data[col] = np.log(data[col])
        else:
            pass
    return data

def plot_skew(data):
    for i in range(data.shape[1]):
        plt.hist(data.iloc[:,i])
        plt.title(data.columns[i])
        plt.show()

def stack_data(data, cat, my, c=np.array([None])):
    if c.any():
        data = np.column_stack([data, cat, my, c])
    else:
        data = np.column_stack([data, cat,my])
    return data


with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    car_data = car_name_fix(car_data)
    car_data = group_company(car_data)
    car_data = hp_fix(car_data)
    car_data = num_cat(car_data)

    car_data, cat_car_data = cat_extractor(car_data)
    cat_car_data = genaralize(cat_car_data, 'company')

    cat_car_data = binarize_discret(cat_car_data, 'company')
    cat_car_data = binarize_discret(cat_car_data, 'origin')

    car_data_cd = cylinder_displacement(car_data)
    car_data_cd['model year'] = rescale(car_data_cd[['model year']])
    car_data[['model year', 'cylinders']] = rescale(car_data[['model year', 'cylinders']])

    cylinders = car_data.values[:,1]
    model_year = car_data.values[:,1]

    car_data = car_data.drop(['cylinders', 'model year'], axis=1)
    car_data_cd = car_data_cd.drop(['model year'], axis=1)
    car_data_cd = log_skew(car_data_cd)
    car_data = log_skew(car_data)

    x = car_data.values[:,1:]
    x_cd = car_data_cd.values[:,1:]
    x_cat = cat_car_data.values

    y = car_data.values[:,0]

    x_cd_rs = rescale(x_cd)
    x_rs = rescale(x)
    x_cd_rs = stack_data(x_cd_rs, x_cat, model_year[:,None])
    x_rs = stack_data(x_rs, x_cat, model_year[:,None], cylinders[:,None])

    x_cd_std = standardize(x_cd)
    x_std = standardize(x)
    x_cd_std = stack_data(x_cd_std, x_cat, model_year[:,None])
    x_std = stack_data(x_std, x_cat, model_year[:,None], cylinders[:,None])

    x = stack_data(x,x_cat,model_year,cylinders)
    x_cd = stack_data(x_cd, x_cat, model_year)

    x_list = [('Data original', x), ('Data original [cd]', x_cd), ('Data Standardized', x_std), ('Data Standardized [cd]', x_cd_std), ('Data Rescaled', x_rs), ('Data Rescaled [cd]', x_cd_rs)]
