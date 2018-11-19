from data import car_data
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def data_peek():
    print(car_data.head(20))

def data_shape():
    print('\nDataset shape:', car_data.shape)

def data_description():
    print('\n',car_data.describe())

def data_correlation():
    print('\nDataCorrelation:\n', car_data.corr())

def data_types():
    print('\n', car_data.dtypes)

def data_nullvalues():
    print('\nNull valuesin Dataset:', car_data.isnull().sum())

def data_skew():
    print('\nSkew in data:\n', car_data.skew())

def plot_data():
    car_data.hist()
    plt.plot()
    plt.show()

def data_group(group):
    print(f'\nThere are {len(np.unique(car_data[group]))} unique values in {group}.')

def find_object():
    values = []
    for value in car_data['horsepower'].values:
        try:
            type(float(value)) == float
        except:
            values.append(value)
    print(f'\nThere are {len(values)} {type(value)} values in horsepower:', values)

def plot_year():
    df = car_data.iloc[:,[0,6]]
    y = df.groupby('model year').mean().values
    x = np.unique(car_data['model year'])
    plt.plot(x,y)
    plt.title('mpg improvement over the years')
    plt.xlabel('years')
    plt.ylabel('mpg [mean]')
    plt.show()

if __name__ == '__main__':
    data_peek()
    data_group('car name')
    data_shape()
    data_description()
    data_correlation()
    data_types()
    data_nullvalues()
    data_skew()
    find_object()
    plot_data()
    plot_year()
