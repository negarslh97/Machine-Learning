# House Price Prediction Pipeline Project

## Table of Contents
1. [Overview](#overview)
2. [Input Files](#input-files)
3. [Output Files](#output-files)
4. [Usage](#usage)
   - [Prerequisites](#prerequisites)
   - [Steps to Run](#steps-to-run)
5. [Data Preprocessing](#data-preprocessing)
   - [Numerical Features](#numerical-features)
   - [Categorical Features](#categorical-features)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Output Explanation](#output-explanation)
9. [Notes on Customization](#notes-on-customization)
10. [Results](#results)
    - [Model Performance](#model-performance)
    - [Correlation Analysis](#correlation-analysis)

## Overview
This repository contains a pipeline for training a regression model and making predictions on a housing dataset. The script is implemented in `main.ipynb`. It trains a model to predict the `SalePrice` (target variable) using numerical and categorical features provided in the dataset.

### Input Files:
- **H_train.csv**: Training dataset containing features and the target column (`SalePrice`).
- **H_test.csv**: Test dataset containing only features without the target column.

### Output Files:
- **model.h5**: The trained regression model saved in HDF5 format.
- **H_test_prediction.csv**: Predictions for the test dataset, including the `SalePrice` values.

## Usage
### Prerequisites
1. Python 3.x
2. Required libraries (install using `pip install -r requirements.txt`):
   - pandas
   - numpy
   - sklearn
   - tensorflow/keras

### Steps to Run
1. Open the `main.ipynb` notebook in Jupyter Notebook or Jupyter Lab.
2. Ensure the input files (`H_train.csv` and `H_test.csv`) are in the same directory as the notebook.
3. Execute the notebook cells sequentially.
4. The following output files will be generated:
   - Trained model saved as `model.h5`
   - Test predictions saved as `H_test_prediction.csv`

## Data Preprocessing
### Numerical Features
1. **Correlation Analysis**: Each numerical feature's correlation with `SalePrice` was computed to identify strong predictors.
2. **Feature Scaling**: Numerical features were scaled using StandardScaler to normalize their distributions, ensuring the model performs well.

### Categorical Features
1. **Encoding**: Categorical variables were transformed into numerical representations using one-hot encoding.
2. **Handling Missing Data**: Missing values in categorical columns were imputed with the most frequent value in each column.

## Model Architecture
1. **Model Type**: A neural network model was implemented using Keras/TensorFlow.
2. **Layers**:
   - Input layer with dimensions matching the processed feature space.
   - Two dense hidden layers with ReLU activation functions.
   - Output layer with a single neuron for regression (linear activation).
3. **Optimizer**: Adam optimizer with a learning rate of 0.001.
4. **Loss Function**: Mean Squared Error (MSE) for regression.

## Training
1. **Train/Test Split**: The `H_train.csv` data was split into a training set and validation set (80/20 split).
2. **Epochs**: The model was trained for 50 epochs with a batch size of 32.
3. **Early Stopping**: Monitored validation loss to prevent overfitting.

## Output Explanation
- **model.h5**: The trained model, which can be reloaded for inference or further training.
- **H_test_prediction.csv**: Contains two columns:
  - `Id`: Identifier for the test sample.
  - `SalePrice`: Predicted sale price for the respective sample.

## Notes on Customization
- To modify preprocessing for numerical or categorical data, update the relevant preprocessing cells in `main.ipynb`.
- To adjust the model architecture, modify the model definition in the corresponding cell.

## Results
### Model Performance
- Evaluate the model performance on the validation set (MSE, RMSE, etc.).
<br/>
<br/>
![Loss](https://github.com/negarslh97/Machine-Learning/blob/main/6.10.Assignment/HousePrices/output/output.png)

### Correlation Analysis
- Key numerical predictors identified from the correlation with `SalePrice` are logged in the notebook.
