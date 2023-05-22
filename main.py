# SAMME implementation using 3 classes from the MNIST dataset
# Inspired by the SAMME article (Multi-class AdaBoost by J. Zhu, H. Zou, S. Rosset, T. Hastie, 2009)

from keras.datasets import mnist
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Total number of classes
# Must be defined apriori
# Cannot be bigger than 10 (restriction applied by the MNIST dataset)
no_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

mask = np.logical_or.reduce([y_train == i for i in range(no_classes)])
x_train = x_train[mask]
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train[mask]

# For final accuracy
mask = np.logical_or.reduce([y_test == i for i in range(no_classes)])
x_test = x_test[mask]
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test[mask]

print('Working with the following matrix shapes:\n')
print(f'x_train : {x_train.shape}')
print(f'y_train : {y_train.shape}')
print(f'x_test : {x_test.shape}')
print(f'y_test : {y_test.shape}')

# Number of training samples
n = y_train.shape[0]

# Starting weights
weights = np.ones(shape=(n)) / n

# Total number of iterations performed
no_iterations = 100

errors = []
alphas = []
classifiers = []

for i in range(1, no_iterations+1):
    print(f'Iteration {i}')

    # Compute the "best" classifier
    current_classifier = DecisionTreeClassifier(max_depth=1)
    current_classifier.fit(x_train, y_train, sample_weight=weights)

    predicted_y_train = current_classifier.predict(x_train)

    # Compute the error
    # e_m = \sum_{i=1}^{n} w_{i} * 1_{c_{i} != T_{m}(x_{i})}
    current_error = np.sum(weights * (y_train != predicted_y_train))

    # Compute the alpha
    current_alpha = np.log((1-current_error) / current_error) + np.log(no_classes - 1)

    # Update the weights
    weights = weights * (np.e ** (current_alpha * (y_train != predicted_y_train)))

    # Re-normalize weights
    weights = weights / np.sum(weights)

    # Retain the computed values
    errors.append(current_error)
    alphas.append(current_alpha)
    classifiers.append(current_classifier)


# Initialize an array to store the weighted class probabilities
weighted_probabilities = np.zeros((len(x_test), no_classes))

# For each classifier and its corresponding alpha
for classifier, alpha in zip(classifiers, alphas):
    
    probabilities = classifier.predict_proba(x_test)

    # Accumulate the weighted probabilities
    weighted_probabilities += alpha * probabilities

# The predicted class is the one with the highest "probability"
prediction_y_test = np.argmax(weighted_probabilities, axis=1)

accuracy = accuracy_score(y_test, prediction_y_test)
print(f'Accuracy: {accuracy:.5f}')
