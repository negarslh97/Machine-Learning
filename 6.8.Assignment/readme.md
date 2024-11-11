# Multi-Layer Perceptron (MLP) for Digit Classification

This repository contains a Python implementation of a simple Multi-Layer Perceptron (MLP) used to classify handwritten digits using the `sklearn` Digits dataset. The model is trained from scratch using NumPy and consists of three fully connected layers with sigmoid activation functions, ending with a softmax activation for classification.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Functions](#functions)
5. [Training](#training)
6. [Usage](#usage)
7. [Model Evaluation](#model-evaluation)


## Requirements

To run this code, you need to install the following Python libraries:

- `numpy`
- `matplotlib`
- `sklearn`

You can install these dependencies via `pip`:

```bash
pip install numpy matplotlib scikit-learn
```

## Dataset

The dataset used in this project is the **Digits Dataset** from the `sklearn.datasets` module. This dataset contains 1,797 8x8 pixel grayscale images of handwritten digits (0-9), with each image flattened into a 64-dimensional feature vector.

## Model Architecture

The architecture of the model is a simple feed-forward neural network with the following layers:

1. **Input Layer**: Accepts 64 features (flattened 8x8 pixel image).
2. **Hidden Layer 1**: 128 neurons with a sigmoid activation function.
3. **Hidden Layer 2**: 32 neurons with a sigmoid activation function.
4. **Output Layer**: 10 neurons (one for each digit class) with a softmax activation function for classification.

## Functions

- **Sigmoid**: Activation function for hidden layers.
- **Softmax**: Activation function for the output layer.
- **Cross-Entropy Loss**: Measures the loss between predicted probabilities and true labels (not used in backpropagation, but could be added for evaluation).
- **Root Mean Squared Error (RMSE)**: Used to measure the performance of the model during training and testing.

## Training

The model is trained for 80 epochs using the following settings:

- Learning rate: `0.001`
- The dataset is split into 80% training and 20% testing.
- The training loop includes both forward propagation and backpropagation.

During training, the following metrics are calculated:

- **Training Loss**: Root mean squared error between predictions and true labels.
- **Training Accuracy**: Percentage of correct predictions on the training set.
- **Test Loss**: Root mean squared error on the test set.
- **Test Accuracy**: Percentage of correct predictions on the test set.

The performance is printed after each epoch.

## Usage

To train the model, simply run the script. It will automatically download the dataset, train the model, and display the loss and accuracy metrics for both the training and test sets after every epoch.

```bash
python main.ipynb
```

### Example Output:
```text
loss train: 0.5234 accuracy train: 0.9378
loss test: 0.6754 accuracy test: 0.9203
...
train completed!
```

## Model Evaluation

After training, the model is evaluated on the test set, and the final test loss and accuracy are displayed. The accuracy values indicate how well the model generalizes to unseen data.
