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
import math

def get_data(filepath):
    return pd.read_csv(filepath)

def init_matrices(data, y_vector, X_columns):
    X = data[X_columns].values
    y = data[y_vector].values
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

def init_theta_vector(theta_len):
    return np.random.random(3)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_func(h, y):
    return -1/len(h) * np.sum(y * np.log(h) + (1-y) * (np.log(1-h)))

def gradient(X, h, y):
    return np.matmul(X.T, (h - y)) / len(h)

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

    theta = init_theta_vector(X.shape[1])
    # print(theta)
    print('Theta vector initialised')

    h = predict(X, theta)
    # print(h)
    print('Calculated predictions')

    J = cost_func(h, y)
    # print(J)
    print('Calculated Cost')

    grad = gradient(X, h, y)
    # print(grad)
    print('Calculated gradient')

    print(theta)

    alpha = 0.05
    for i in range(10000):
        h = predict(X, theta)
        # print(cost_func(h,y))
        for j in range(len(theta)):
            theta[j] = theta[j] - alpha/len(h) * np.sum(np.dot((h-y), (X[:,j])))

    print(theta)
    print(cost_func(h,y))


main()
