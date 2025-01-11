from sklearn.model_selection import train_test_split
from ROCAnalysis import ROCAnalysis

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the TP-rate of a given model.
    """

    def __init__(self, X, y, model):
        """
        Initializes the ForwardSelection object with the dataset and the model.
        """
        self.X = X
        self.y = y
        self.model = model
        self.selected_features = []
        self.best_score = float('-inf')
        self.X_train, self.X_test, self.y_train, self.y_test = self.create_split()

    def create_split(self):
        """
        Splits the data into 80% training and 20% testing.
        """
        return train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def evaluate_model(self, features):
        """
        Trains the model on the selected features and evaluates the TP-rate using the ROCAnalysis class.
        """
        X_train_fs = self.X_train[:, features]
        X_test_fs = self.X_test[:, features]
        
        self.model.fit(X_train_fs, self.y_train)
        predictions = self.model.predict(X_test_fs)
        
        # Instantiate the ROCAnalysis with the predicted and true labels
        roc = ROCAnalysis(predictions, self.y_test)
        return roc.tp_rate()

    def forward_selection(self):
        """
        Performs forward feature selection.
        """
        features = list(range(self.X.shape[1]))
        
        while features:
            best_feature = None
            
            for feature in features:
                current_features = self.selected_features + [feature]
                score = self.evaluate_model(current_features)
                
                if score > self.best_score:
                    self.best_score = score
                    best_feature = feature
            
            if best_feature is not None:
                self.selected_features.append(best_feature)
                features.remove(best_feature)
            else:
                break

    def fit(self):
        """
        Fits the model using the selected features.
        """
        self.forward_selection()
        self.model.fit(self.X_train[:, self.selected_features], self.y_train)

    def predict(self, X):
        """
        Predicts the target labels for the given test features using the selected features.
        """
        X_fs = X[:, self.selected_features]
        return self.model.predict(X_fs)
