import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                distances.append(self.euclidean_distance(x, x_train))

            nearest_neighbors = np.argsort(distances)[:self.k]
            nearest_labels = self.Y_train[nearest_neighbors]
            most_common = np.bincount(nearest_labels).argmax()
            y_pred.append(most_common)  # Append the prediction to y_pred list
        return np.array(y_pred)



    def evaluate(self, X, Y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == Y) / len(Y)
        return accuracy
