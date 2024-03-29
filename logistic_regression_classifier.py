'''
Author: Ben McCoy
Written: Aug 2019

This script will be a logisitic regression classifier based on material
learned in Andrew Ng's Machine Learning course.

It will begin as a simple one class classifier, but I expect to expand it
to be able to handle any number of classes.

I will use data from http://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work
and try to determine the cause of an absence.

The script uses the argparse library to take arugments from the command line.
'''

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import operator

def get_data(filepath):
    return pd.read_csv(filepath)

def init_matrices(data, y_vector, X_columns):
    '''separate the data into an X feature matrix and a y results vector'''
    X = data[X_columns].values
    y = data[y_vector].values
    return X, y

def scale_test_data(X_guess, X, option):
    X_guess_copy = X_guess.copy()
    for col in range(X_guess.shape[1]):
        if option == 'min-max':
            X_guess_copy[:,col] = (X_guess[:,col] - np.min(X[:,col])) / (np.max(X[:,col]) - np.min(X[:,col]))
        elif option == 'mean-norm':
            X_guess_copy[:,col] = (X_guess[:,col] - np.mean(X[:,col])) / (np.max(X[:,col]) - np.min(X[:,col]))
        elif option == 'standardization':
            X_guess_copy[:,col] = (X_guess[:,col] - np.mean(X[:,col])) / (np.std(X[:,col]))

    return X_guess_copy

def scale_features(X, option):
    '''scales features based on the scaling type'''
    X_norm = X.copy()
    for col in range(X_norm.shape[1]):
        if option == 'min-max':
            X_norm[:,col] = (X_norm[:,col] - np.min(X_norm[:,col])) / (np.max(X_norm[:,col]) - np.min(X_norm[:,col]))
        elif option == 'mean-norm':
            X_norm[:,col] = (X_norm[:,col] - np.mean(X_norm[:,col])) / (np.max(X_norm[:,col]) - np.min(X_norm[:,col]))
        elif option == 'standardization':
            X_norm[:,col] = (X_norm[:,col] - np.mean(X_norm[:,col])) / (np.std(X_norm[:,col]))
    return X_norm

def add_ones_column(X, axis_val):
    return np.insert(X, 0, 1, axis=axis_val)

def init_theta_vector(y_labels, theta_len):
    '''creates a pandas DF of random theta values with each row being used
    for a different y_label'''
    columns = []
    for i in range(theta_len):
        columns.append('theta' + str(i))
    theta_df = pd.DataFrame(np.random.rand(), index=y_labels, columns=columns)
    return theta_df

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, theta):
    return sigmoid(np.dot(X, theta))

def cost_func(h, y):
    return -1/len(h) * np.sum(y * np.log(h) + (1-y) * (np.log(1-h)))

def gradient(X, h, y):
    return np.matmul(X.T, (h - y)) / len(h)

def learn_theta(alpha, epochs, X, theta, y, print_status):
    '''progressively learns and improves the values of theta using the Learning
    rate alpha over number of epochs stated. Records the history of the cost
    function to be plotted for debugging'''

    J_hist_dict = {}
    learned_theta = theta.copy()

    for index, row in theta.iterrows():
        theta_vals = row.values
        J_history = []

        y_adjusted = []
        for h in y:
            if h == index:
                y_adjusted.append(1)
            else:
                y_adjusted.append(0)

        y_adjusted = np.array(y_adjusted)

        for i in range(epochs):
            h = predict(X, theta_vals)

            for j in range(len(theta_vals)):
                theta_vals[j] = theta_vals[j] - alpha/len(h) * np.sum(np.dot((h-y_adjusted), (X[:,j])))

            if print_status == True:
                print(cost_func(h,y_adjusted))
            J_history.append(cost_func(h,y_adjusted))

        J_hist_dict['J_hist_' + str(index)] = J_history
        learned_theta.loc[index] = theta_vals #This may need to change based on str index or int indexes

    return learned_theta, J_hist_dict


def plot_cost(J_history, legend):
    plt.plot(range(len(J_history)), J_history, label=str(legend))

def guess(X_guess, learned_theta):
    guess_dict = {}
    for index, row in learned_theta.iterrows():
        theta_vals = row.values
        guess_dict[index] = predict(X_guess, theta_vals)
    print(guess_dict)
    return max(guess_dict.items(), key=operator.itemgetter(1))[0]

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='training data filepath')
    parser.add_argument('y_column', type=str,
                        help='the column title of the outputs column')
    parser.add_argument('X_columns', nargs='+',
                        help='the columns to be included as features in X matrix')
    parser.add_argument('-scaling', type=str, default='standardization',
                        help='sclaing options are: min-max, mean-norm, standardization (default)')
    parser.add_argument('-epochs', nargs='?', type=int, default=5000,
                        help='the number of iterations to run through, default is 5000')
    parser.add_argument('-alpha', nargs='?', type=float, default=0.01,
                        help='the chosen learning rate, default is 0.01')
    parser.add_argument('-test', type=str,
                        help='the testing data filepath')
    parser.add_argument('-pr', '--print',
                        help='prints the cost function after erach iteration',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plots the cost function over time',
                        action='store_true')
    args = parser.parse_args()

    data = get_data(args.data)
    # print(data)
    print('Data Collected')

    # print(args.X_columns)
    X, y = init_matrices(data, args.y_column, args.X_columns)
    # print(X)
    # print(y)
    print('X and y matrices created')

    X_norm = scale_features(X, str(args.scaling))
    # print(X)
    print('Features scaled')

    X_norm = add_ones_column(X_norm, 1)
    # print(X)
    print('Ones column added')

    theta = init_theta_vector(list(set(y)), X_norm.shape[1])
    # print(theta)
    print('Theta vector initialised')

    learned_theta, J_hist_dict = learn_theta(float(args.alpha), args.epochs, X_norm, theta, y, args.print)
    # print(learned_theta)
    print('Theta values learned and J_history saved')

    if args.plot:
        for key, value in J_hist_dict.items():
            plt.plot(range(len(value)), value, label=str(key))
        plt.title('Cost Function Ouput Vs Iteration Number')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function Output')
        plt.legend()
        plt.show()

    if args.test:
        test_data = get_data(str(args.test))
        X_guess = test_data[args.X_columns].values
        X_guess_scaled = scale_test_data(X_guess, X, str(args.scaling))
        X_guess_scaled = add_ones_column(X_guess_scaled, 1)
        print(X_guess_scaled)
        for index, row in test_data.iterrows():
            y_guess = guess(X_guess_scaled[index], learned_theta)
            test_data.loc[index, args.y_column] = y_guess
        print(test_data)

main()
