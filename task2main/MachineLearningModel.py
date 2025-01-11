from abc import ABC, abstractmethod
import numpy as np

class MachineLearningModel(ABC):
    """
    Abstract base class for machine learning models.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using the given training data.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted values.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y):
        """
        Evaluate the model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score.
        """
        pass

def _polynomial_features(self, X):
    """
        Generate polynomial features from the input features.
        Check the slides for hints on how to implement this one. 
        This method is used by the regression models and must work
        for any degree polynomial
        Parameters:
        X (array-like): Features of the data.

        Returns:
        X_poly (array-like): Polynomial features.
    """
    degree = self.degree
    samples, features = X.shape
    X_poly = np.ones((samples, 1)) 
    for d in range(1, degree + 1):
        for j in range(features):
            X_poly = np.concatenate((X_poly, np.power(X[:, j], d)[:, np.newaxis]), axis=1)
    return X_poly

class RegressionModelNormalEquation(MachineLearningModel):
    def __init__(self, degree):
        self.degree = degree
        self.theta = None  
        self.cost = []  

    def fit(self, X, y):
        Xpoly = _polynomial_features(self, X)
        self.theta = np.linalg.pinv(Xpoly.T.dot(Xpoly)).dot(Xpoly.T).dot(y) 
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        self.cost.append(mse)

    def predict(self, X):
        X_poly = _polynomial_features(self, X)
        predictions = np.dot(X_poly, self.theta)
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

class RegressionModelGradientDescent:
    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        super().__init__()
        self.degree = degree
        self.learning_rate = learning_rate 
        self.num_iterations = num_iterations 
        self.theta = None
        self.cost = []

    def fit(self, X, y):
        X_poly = _polynomial_features(self, X)
        self.theta = np.zeros(X_poly.shape[1])
        for _ in range(self.num_iterations):
            predictions = self.predict(X)
            errors = np.dot(X_poly, self.theta) - y
            gradients = np.dot(X_poly.T, errors) / len(y)
            self.theta -= self.learning_rate * gradients
            mse = np.mean((predictions - y) ** 2)
            self.cost.append(mse)

    def predict(self, X):
        X_poly = _polynomial_features(self, X)
        predictions = np.dot(X_poly, self.theta)
        return predictions

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None  
        self.cost_history = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        m = len(y)
        predictions = self._sigmoid(np.dot(X, self.weights))
        cost = -1 / m * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n + 1)  
        X = np.hstack([np.ones((m, 1)), X])  
        for _ in range(self.num_iterations):
            predictions = self._sigmoid(np.dot(X, self.weights))
            error = predictions - y
            gradient = np.dot(X.T, error) / m
            self.weights -= self.learning_rate * gradient  
            self.cost_history.append(self._cost_function(X, y))

    def predict(self, X):
        m = X.shape[0]
        X = np.hstack([np.ones((m, 1)), X])  
        probabilities = self._sigmoid(np.dot(X, self.weights))
        return (probabilities >= 0.5).astype(int)  

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    @property
    def theta(self):
        return self.weights  

    @theta.setter
    def theta(self, value):
        self.weights = value  
    
class NonLinearLogisticRegression:
    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.cost_history = []  

    def mapFeature(self, X1, X2):
        degree = self.degree
        out = [np.ones(len(X1))]  
        for i in range(1, degree + 1):
            for j in range(i + 1):
                out.append((X1 ** (i - j)) * (X2 ** j))
        return np.column_stack(out)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        predictions = self.predict(X)
        logpre = np.log(predictions)
        cost = -np.mean(y * logpre + (1 - y) * np.log(1 - predictions))
        return cost

    def fit(self, X, y):
        X1, X2 = X[:, 0], X[:, 1]
        X_poly = self.mapFeature(X1, X2)
        num_features = np.size(X_poly, 1)  
        self.weights = np.zeros(num_features)

        for _ in range(self.num_iterations):
            z = np.dot(X_poly, self.weights)
            h = self._sigmoid(z)
            gradient = np.dot(X_poly.T, (h - y)) / len(y)
            self.weights -= self.learning_rate * gradient
            cost = self._cost_function(X_poly, y)
            self.cost_history.append(cost)  

        return self

    def predict(self, X):
        X1, X2 = X[:, 0], X[:, 1]
        X_poly = self.mapFeature(X1, X2)
        probabilities = self._sigmoid(np.dot(X_poly, self.weights))
        return probabilities

    def evaluate(self, X, y):
        cost = self._cost_function(X, y)
        return cost

    @property
    def theta(self):
        return self.weights  

    @theta.setter
    def theta(self, value):
        self.weights = value  