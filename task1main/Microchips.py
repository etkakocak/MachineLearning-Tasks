# Exercise 1: k-NN Classification

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
from matplotlib.colors import ListedColormap

# To calculate the Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# k-NN classifier
class KNN:
    # Initialize the KNN classifier with a specific k value (number of neighbors)
    def __init__(self, k=3):
        self.k = k

    # Stores the dataset, as k-NN is a lazy learner
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Applies the _predict function to each example in X
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # Predicts the label of a single data point using its k nearest neighbors
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = mode(k_nearest_labels).mode[0]
        return most_common

# Plot the original microchip data
def plot_original_data(X, y):
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', marker='x', label='Fail')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', marker='o', label='OK')
    plt.title('Microchip Tests')
    plt.xlabel('Test 1')
    plt.ylabel('Test 2')
    plt.legend()
    plt.show()

# Plot the decision boundary and also the training points
def plot_decision_boundary(clf, X, y, k, subplot):
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    mesh_step_size = .01  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    subplot.contourf(xx, yy, Z, cmap=cmap_light)

    subplot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    subplot.set_xlim(xx.min(), xx.max())
    subplot.set_ylim(yy.min(), yy.max())
    subplot.set_title(f"k = {k}, Training Errors = {int((clf.predict(X) != y).sum())}")

# Load the dataset
data = pd.read_csv('datasets/microchips.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

plot_original_data(X, y)

k_values = [1, 3, 5, 7]
unknown_microchips = np.array([[-0.3, 1.0], [-0.5, -0.1], [0.6, 0.0]])

# Predictions for unknown microchips at different k values
predictions = []
for k in k_values:
    knn = KNN(k=k)
    knn.fit(X, y)
    predictions_for_k = []
    print(f"k = {k}")
    for i, chip in enumerate(unknown_microchips, start=1):
        prediction = knn.predict([chip])[0]
        predictions_for_k.append(('OK' if prediction == 1 else 'Fail'))
        print(f"chip{i}: {chip} ==> {'OK' if prediction == 1 else 'Fail'}")
    predictions.append(predictions_for_k)

plt.tight_layout()