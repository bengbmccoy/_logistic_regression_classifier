'''
Author: Ben McCoy
Written: Aug 2019

This script will be a logisitic regression classifier based on material
learned in Andrew Ng's Machine Learning course.

It will begin as a simple one class classifier, but I expect to expand it
to be able to handle any number of classes.

I will use data from http://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
and try to determine the cause of an absence
'''

import pandas as pd
import numpy as np

def get_data(filepath):
    return pd.read_csv(filepath)

def init_matrices(data, y_vector, X_columns):
    X = data[X_columns].values
    y = np.vstack(data[y_vector].values)
    return X, y

def scale_features(X, option):
    X_norm = X
    for col in range(X_norm.shape[1]):
        if option == 'min-max':
            X_norm[:,col] = (X_norm[:,col] - np.min(X_norm[:,col])) / (np.max(X_norm[:,col]) - np.min(X_norm[:,col]))
        elif option == 'mean-norm':
            X_norm[:,col] = (X_norm[:,col] - np.mean(X_norm[:,col])) / (np.max(X_norm[:,col]) - np.min(X_norm[:,col]))
        elif option == 'standardization':
            X_norm[:,col] = (X_norm[:,col] - np.mean(X_norm[:,col])) / (np.std(X_norm[:,col]))
    return X_norm

def add_ones_column(X):
    return np.insert(X, 0, 1, axis=1)

def main():

    data = get_data('simple_training.csv')
    # print(data)
    print('Data Collected')

    X, y = init_matrices(data, 'Y', ['X1', 'X2'])
    # print(X)
    # print(y)
    print('X and y matrices created')

    X = scale_features(X, 'min-max')
    # print(X)
    print('Features scaled')

    X = add_ones_column(X)
    # print(X)
    print('Ones column added')



main()
