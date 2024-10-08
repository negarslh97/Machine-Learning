import numpy as np
import numpy.linalg as inv

class LLS:
    def __init__(self):
        self.w = None

    def fit(self, X_train, y_train):
        self.w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    def predict(self, X_test):
        y_pred = X_test @ self.w
        return y_pred
    
    def evaluate(self, X_test, y_test, metric):
        y_pred = self.predict(X_test)

        if metric == "mae":
            loss = np.sum(np.abs(y_test - y_pred)) / len(y_test)
        elif metric == "mse":
            loss = np.sum((y_test - y_pred) ** 2) / len(y_test)
        elif metric == "rmse":
            loss = np.sqrt(np.sum((y_test - y_pred) ** 2) / len(y_test))
        else:
            loss = None
            
        return loss
