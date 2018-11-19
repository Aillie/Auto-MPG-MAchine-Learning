import pandas as pd


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin', 'car name']
car_data = pd.read_csv(url, names=names, delim_whitespace=True)
