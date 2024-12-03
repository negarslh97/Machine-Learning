# Machine Learning Projects

## Table of Contents
1. [MLP and Perceptron Project](#mlp-and-perceptron-project)
   - [Project Overview](#project-overview-1)
   - [Prerequisites](#prerequisites-1)
   - [Project Structure](#project-structure)
   - [Technologies and Libraries](#technologies-and-libraries-1)
   - [How to Run](#how-to-run-1)
   - [Results and Analysis](#results-and-analysis-1)

2. [House Price Prediction Pipeline Project](#house-price-prediction-pipeline-project)
   - [Overview](#overview)
   - [Prerequisites](#prerequisites-2)
   - [Steps to Run](#steps-to-run)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architecture](#model-architecture)
   - [Training](#training)
   - [Output Explanation](#output-explanation)
   - [Results](#results-1)

---

## MLP and Perceptron Project

### Project Overview
This project focuses on implementing and comparing two machine learning models: **Multilayer Perceptron (MLP)** and **Perceptron**. The objective is to analyze their performance in predicting and classifying input data. The notebook includes preprocessing steps, model training, and a detailed comparison of the models.

### Prerequisites
To run this project, you will need:
- **Python 3.7 or higher**
- An environment such as **Google Colab** or **Jupyter Notebook**

### Project Structure
The project is divided into the following sections:
1. **Preprocessing**:
   - Loading and cleaning the dataset.
   - Splitting the data into training and testing sets.
2. **Modeling**:
   - Implementing the MLP model using TensorFlow.
   - Developing the Perceptron model.
3. **Comparison**:
   - Evaluating and comparing the models using various metrics.

### Technologies and Libraries
- `numpy` and `pandas` for data manipulation and processing
- `matplotlib` for data visualization
- `tensorflow` for building and training neural networks
- `scikit-learn` for data splitting and evaluation metrics
- `tabulate` for displaying tabular data

### How to Run
1. Clone the repository or download the project files.
2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib tensorflow scikit-learn tabulate
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```
4. Follow the steps in the notebook for preprocessing, training, and model comparison.

### Results and Analysis
In this project:
- The performance of **MLP** and **Perceptron** models was evaluated using metrics such as accuracy, precision, and recall.
- **MLP** generally performs better on complex datasets, whereas **Perceptron** is suitable for simpler problems.

![Temperature of Days - MLP](https://github.com/negarslh97/Machine-Learning/blob/main/6.10.Assignment/Weather/output/mlp.png)

<p align="center">
  <img src="https://github.com/negarslh97/Machine-Learning/blob/main/6.10.Assignment/Weather/output/loss_mlp.png" alt="MLP Loss" width="45%">
  <img src="https://github.com/negarslh97/Machine-Learning/blob/main/6.10.Assignment/Weather/output/loss_perceptron.png" alt="Perceptron Loss" width="45%">
</p>

| Model                         | Loss    |
|-------------------------------|---------|
| Multi Layer Perceptron        | 13.3313 |
| Single (One) Layer Perceptron | 90.1282 |

---

## House Price Prediction Pipeline Project

### Overview
This repository contains a pipeline for training a regression model and making predictions on a housing dataset. The script is implemented in `main.ipynb`. It trains a model to predict the `SalePrice` (target variable) using numerical and categorical features provided in the dataset.

### Prerequisites
1. Python 3.x
2. Required libraries :
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

### Data Preprocessing
#### Numerical Features
1. **Correlation Analysis**: Each numerical feature's correlation with `SalePrice` was computed to identify strong predictors.
2. **Feature Scaling**: Numerical features were scaled using StandardScaler to normalize their distributions, ensuring the model performs well.

#### Categorical Features
1. **Encoding**: Categorical variables were transformed into numerical representations using one-hot encoding.
2. **Handling Missing Data**: Missing values in categorical columns were imputed with the most frequent value in each column.

### Model Architecture
1. **Model Type**: A neural network model was implemented using Keras/TensorFlow.
2. **Layers**:
   - Input layer with dimensions matching the processed feature space.
   - Two dense hidden layers with ReLU activation functions.
   - Output layer with a single neuron for regression (linear activation).
3. **Optimizer**: Adam optimizer with a learning rate of 0.001.
4. **Loss Function**: Mean Squared Error (MSE) for regression.

### Training
1. **Train/Test Split**: The `H_train.csv` data was split into a training set and validation set (80/20 split).
2. **Epochs**: The model was trained for 50 epochs with a batch size of 32.
3. **Early Stopping**: Monitored validation loss to prevent overfitting.

### Output Explanation
- **model.h5**: The trained model, which can be reloaded for inference or further training.
- **H_test_prediction.csv**: Contains two columns:
  - `Id`: Identifier for the test sample.
  - `SalePrice`: Predicted sale price for the respective sample.

### Results
#### Model Performance
- Evaluate the model performance on the validation set (MSE, RMSE, etc.).

<br/>

![Loss](https://github.com/negarslh97/Machine-Learning/blob/main/6.10.Assignment/HousePrices/output/output.png)

#### Correlation Analysis
- Key numerical predictors identified from the correlation with `SalePrice` are logged in the notebook.

