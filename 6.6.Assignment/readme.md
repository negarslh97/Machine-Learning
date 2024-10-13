# Table of Contents

- [Overview](#overview)
- [Structure](#structure)
  - [1. Linear Dataset Creation](#1-linear-dataset-creation)
  - [2. Dataset Splitting](#2-dataset-splitting)
  - [3. Perceptron Algorithm](#3-perceptron-algorithm)
  - [4. Employee's Salary Dataset](#4-employees-salary-dataset)
  - [5. Visualizing Results (Graphs)](#5-visualizing-results-graphs)
  - [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
  - [7. Abalone Dataset](#7-abalone-dataset)
  - [8. Boston House-Prices Dataset](#8-boston-house-prices-dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Instructions](#instructions)


# Overview
This project implements a Perceptron algorithm to solve regression problems using multiple datasets such as Employee Salary, Abalone, and Boston House-Prices datasets. The perceptron algorithm is implemented from scratch in Python, and the project also includes visualization techniques for both data and model performance metrics like loss.

The key features covered in the project are:

Linear Regression Dataset Creation using Scikit-learn's make_regression function.
Perceptron Algorithm implemented as a class for solving regression tasks.
Fitting the Model on different datasets.
Visualizing Data and Loss Graphs for performance insights.
Tuning Hyperparameters to achieve the best results.
3D Visualization and Animation of the regression solution for the Boston House-Prices dataset.

# Structure
## 1. Linear Dataset Creation
The project starts by generating a synthetic linear regression dataset using the Scikit-learn library's make_regression function:

```python
from sklearn.datasets import make_regression
```

This dataset simulates real-world data, and it is used to initially test the perceptron model's performance.

## 2. Dataset Splitting
All datasets are split into train and test sets to evaluate the model's performance:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

## 3. Perceptron Algorithm
A custom Perceptron algorithm is implemented from scratch as a Python class. This class is responsible for:

Initializing weights and biases.
Forward propagation for prediction.
Backward propagation for calculating gradients and updating weights.
Handling loss calculation using Mean Squared Error.
The model is then fitted to the datasets, and the learned parameters are used to make predictions.

## 4. Employee's Salary Dataset
The **employee salary dataset** is used as an initial dataset to fit the perceptron model. The algorithm aims to predict salaries based on various features provided in the dataset.

<br/>

![correlation](https://github.com/negarslh97/Machine-Learning/blob/main/6.5.Assignment/Tehran_House_Price/output/correlation.png)

<br/>

## 5. Visualizing Results (Graphs)
After training the model on each dataset, the following plots are created:

- Data Graph: Showing how the model fits the training data.
- Loss Graph: Plotting the loss over epochs to monitor the convergence of the perceptron.

The plots are organized as subplots in one window for better comparison.

```python
import matplotlib.pyplot as plt
# Example for plotting data and loss
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(X_train, y_train)
ax2.plot(loss_values)
plt.show()
```

<br/>

![loss](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Employee's_Salary/output/output2.png)

<br/>
<br/>

![loss](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Employee's_Salary/output/output3.png)

<br/>

## 6. Hyperparameter Tuning
Hyperparameters such as learning rate and number of epochs are adjusted to optimize the model performance. This is done across different datasets to achieve the best results.

## 7. Abalone Dataset
The Abalone dataset is another dataset used to test the perceptron. After fitting the model, similar visualizations (data and loss graphs) are generated for evaluation.

<br/>

![correlation](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Abalone/output/output1.png)

<br/>

<br/>

![loss](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Abalone/output/output2.png)

<br/>

## 8. Boston House-Prices Dataset
For the Boston House-Prices dataset, the following steps are performed:

Load the dataset using Scikit-learn:

```python
from sklearn.datasets import fetch_openml

boston = fetch_openml(name="boston", version=1, as_frame=True)
```
● With the help of our correlation choose two features (e.g., "RM" and "LSTAT") as inputs (X), and the price as the target (Y).

<br/>

![correlation](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Boston_House_Prices/output/output.png)

<br/>

● The perceptron algorithm is applied to solve the regression problem using these two features.

●  A 3D plot is generated where the predicted values (`y_pred`) are displayed as a plane, with animation showing how the model fits the data over time. A separate class, `PerceptronVisualizer`, handles the 3D visualization and animation. This class:

- Visualizes training data, test data, and predictions.
- Creates a 3D surface for the perceptron's predictions.
- Animates the fitting process by updating the surface dynamically.

The main script calls this class to manage all 3D plotting and animation tasks, making the code more modular and reusable.

<br/>

![correlation](https://github.com/negarslh97/Machine-Learning/blob/main/6.6.Assignment/Boston_House_Prices/output/perceptron_animation.gif)

<br/>

# Requirements

Libraries
The following libraries are required for running the code:
- `numpy`
- `matplotlib`
- `scikit-learn`

# Installation
To install the necessary libraries, run:

```bash
pip install numpy matplotlib scikit-learn
```

# Instructions

1. Run the main script to test the perceptron on different datasets.
2. Adjust hyperparameters in the perceptron class (learning rate, epochs) to tune the model.
3. Visualize the results using the graphs generated during the training process.
4. 3D Plot and Animation for the Boston House-Prices dataset will display a plane showing the predicted surface based on the chosen features.