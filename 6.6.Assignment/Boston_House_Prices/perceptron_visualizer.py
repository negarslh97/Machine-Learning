import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

class PerceptronVisualizer:
    def __init__(self, model, X_train, y_train, X_test, y_test, y_pred):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.X1_grid, self.X2_grid, self.y_surface = self.create_surface()

    def plot_data(self):
        self.ax.scatter(self.X_train[:, 0], self.X_train[:, 1], self.y_train, color='#a2d2ff', label='Training data', alpha=0.7)
        self.ax.scatter(self.X_test[:, 0], self.X_test[:, 1], self.y_test, color='#ffafcc', label='Test data', alpha=0.7)
        self.ax.scatter(self.X_test[:, 0], self.X_test[:, 1], self.y_pred, color='#c1121f', label='Predictions', alpha=0.3)

    def setup_plot(self):
        self.ax.set_xlabel('RM')
        self.ax.set_ylabel('LSTAT')
        self.ax.set_zlabel('Price')
        self.ax.set_title('3D Perceptron Predictions')
        self.ax.legend()

    def create_surface(self):
        x1_range = np.linspace(self.X_train[:, 0].min(), self.X_train[:, 0].max(), 30)
        x2_range = np.linspace(self.X_train[:, 1].min(), self.X_train[:, 1].max(), 30)
        X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
        X_grid = np.c_[X1_grid.ravel(), X2_grid.ravel()]
        y_surface = self.model.predict(X_grid).reshape(X1_grid.shape)
        return X1_grid, X2_grid, y_surface

    def update(self, frame):
        self.ax.cla() 
        self.plot_data()  
        y_surface_updated = self.y_surface + (frame - 15) * 0.5 
        self.ax.plot_surface(self.X1_grid, self.X2_grid, y_surface_updated, color='orange', alpha=0.5)
        self.setup_plot() 

    def animate(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=30, interval=200)
        plt.show()
        return ani 
