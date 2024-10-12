import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('input/weight-height.csv')

X = df['Height'].values
y = df['Weight'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.8)

X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

fig, (ax1,ax2) = plt.subplots(1,2)


w = np.random.rand(1,1)
b = np.random.rand(1,1)

learning_rate_w = 0.0001
learning_rate_b = 0.1

losses = []

for j in range(50):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = y_train[i]

        y_pred = x @ w + b 
        error = y - y_pred

        # SGD update 
        w = w + (error * x * learning_rate_w)
        w = w + (error * learning_rate_b)
        
        # MAE
        # loss = np.mean(np.abs(error))
        # losses.append(loss)

        # MSE
        loss = np.mean(error ** 2)
        losses.append(loss)

        Y_pred = X_train @ w + b

        ax1.clear()
        ax1.scatter(X_train,y_train,c='blue')
        ax1.plot(X_train,Y_pred,c='red')

        ax2.clear()
        ax2.plot(losses)

        plt.pause(0.01)
