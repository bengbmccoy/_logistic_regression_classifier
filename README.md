# _logistic_regression_classifier
A logistic regression classifier for general use

This logistic regression classifier is expected to be a "off the shelf" usable
script that can generally be helpful in future use.

Thanks to Andrew Ng and Stanford Uni, as this logistic regression classifier
is based on material taught in Andrew Ng's Machine Learning course.

The script takes the following positional command line arguments:

- data - the filepath to the training data set
- y_column - the column of the training data that is the output class
- x_columns - at least one column to be used as input features for the classifier

The script takes the following optional command line arguments:

- scaling - allows the user to choose between min-max, mean-norm and standardization (default)
- epochs - the number of learning iterations (default is 5000)
- alpha - the chosen learning rate (default is 0.01)
- test - the file path for the test data set
- print - an option to print the cost function results after each iteration
- plot - an option to plot the results of the cost function vs the number of iterations

An example of the usage in the command line:

- python .\logistic_regression_classifier.py .\simple_training.csv Y X1 X2

- python .\logistic_regression_classifier.py .\train_iris.csv class sepal_len sepal_wid petal_len petal_wid -pr -p -test .\test_iris.csv

Example training dataset:

In this repo I have used the iris flower training data set that is a common and readily available online. I would like to thank the University of California, Irving for making this data publicly available. Below is the link to the dataset:
https://archive.ics.uci.edu/ml/datasets/iris

The data set contains 150 individual flowers and their species, sepal length, sepal width, petal length and petal width.

TODO:
- Add comments and double check variables to make sure that they are consistent
and they are not duplicative/unneeded
