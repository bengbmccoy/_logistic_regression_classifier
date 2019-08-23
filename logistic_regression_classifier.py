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

def get_data(filepath):
    return pd.read_csv(filepath)

def main():

    print('hello world')

    data = get_data('simple_training.csv')
    print data

main()
