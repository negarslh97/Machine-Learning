# Titanic Survival Prediction Project

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Preprocessing](#preprocessing)
4. [Model Implementation](#model-implementation)
    - [Neural Network](#neural-network)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Perceptron](#perceptron)
5. [Results](#results)
6. [How to Run](#how-to-run)

---

## Introduction
This project predicts survival on the Titanic using machine learning techniques. We preprocess the data and train multiple models to compare their performance.

---

## Dataset Description
The datasets used in this project include:
- `titanic_train.csv`: Training data
- `titanic_test.csv`: Testing data
- `titanic_gender_submission.csv`: Expected results for evaluation

### Features:
- **Pclass**: Passenger class (1st, 2nd, 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age in years
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## Preprocessing
1. **Handle Missing Data**: Replace missing values in `Age` and `Fare` with the median.
2. **Feature Engineering**: 
    - Convert categorical variables (`Sex`, `Embarked`) to numerical values.
    - Create new features:
        - `FamilySize`: Total number of family members aboard.
        - `IsAlone`: Indicator of whether the passenger is alone.
3. **Scaling**: Standardize numerical features using `StandardScaler`.

---

## Model Implementation

### Neural Network
A Multi-Layer Perceptron (MLP) with the following architecture:
- **Input Layer**: 6 nodes
- **Hidden Layers**:
  - Layer 1: 20 neurons, `ReLU` activation
  - Layer 2: 8 neurons, `Sigmoid` activation
- **Output Layer**: 2 neurons, `Softmax` activation
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

### K-Nearest Neighbors (KNN)
KNN with 5 neighbors was implemented to predict survival. The model was trained on the same preprocessed data.

### Perceptron
A simple perceptron classifier was trained as an additional baseline for comparison.

---

## Results

| Algorithm                     | Accuracy |
|-------------------------------|----------|
| KNN                           | **0.83** |
| Perceptron                    | **0.76** |
| MLP (Multi-Layer Perceptron)  | **0.87** |

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/titanic-prediction.git
   cd titanic-prediction
