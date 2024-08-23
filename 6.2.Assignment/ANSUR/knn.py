# import numpy as np


# class KNN:
#     def __init__(self, k):
#         self.k = k

#     def fit(self, X, Y):
#         self.X_train = X
#         self.Y_train = Y

#     def euclidean_distance(self, x1, x2):
#         return np.sqrt(np.sum((x1 - x2) ** 2))

#     def predict(self, list_x):
#         y_list = []
#         for x in list_x:
#             distances = []
#             for x_train in self.X_train:
#                 distances.append(self.euclidean_distance(x, x_train))

#             nearest_neighbors = np.argsort(distances)[:self.k]
#             result = np.bincount(self.Y_train[nearest_neighbors])
#             y = np.argmax(result)
#             y_list.append(y)
#         return y_list

#     def evaluate(self, X, Y):
#         y_pred = self.predict(X)
#         correct_count = 0
#         total_count = len(Y)

#         for pred, actual in zip(y_pred, Y):
#             if pred == actual:
#                 correct_count += 1

#         accuracy = correct_count / total_count
#         return accuracy

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