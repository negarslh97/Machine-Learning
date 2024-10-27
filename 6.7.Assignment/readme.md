# Table of Contents

- [Overview](#overview)
- [Surgical Data Classification 🏨](#surgical-data-classification-🏨)
  - [Project Overview](#project-overview)
  - [Implementation Steps](#implementation-steps)
  - [Plots](#plots)
- [Weather Prediction 🌦️](#weather-prediction-🌦️)
  - [Project Overview](#project-overview-1)
  - [Implementation Steps](#implementation-steps-1)
  - [Important Plots During Training](#important-plots-during-training)
- [Note](#note)

---

# Overview
This repository contains two main machine learning projects utilizing the Perceptron algorithm: one for classifying surgical data and another for predicting weather temperatures. Each project uses supervised learning techniques, and the Perceptron model is applied linearly on both datasets.

---

## Surgical Data Classification 🏨

### Project Overview
In this project, the goal is to classify binary-labeled surgical data using an object-oriented Perceptron algorithm.

### Implementation Steps
1. **Training the Perceptron**: Implemented a Perceptron model and applied it to the surgical dataset for binary classification.
2. **Experimenting with Activation Functions**: Five different activation functions were tested, including `sigmoid`, `tanh`, `ReLU`, `linear`, and others.
3. **Accuracy and Loss Visualization**:
   - Plotted accuracy and loss metrics for both training and test datasets for each epoch.

### Plots
- **Accuracy Plot**: Displays model accuracy over each training epoch for both training and test datasets.
- **Loss Plot**: Shows loss over each training epoch for both training and test datasets.

<br/>
<br/>
![Evaluate Plot](https://github.com/negarslh97/Machine-Learning/blob/main/6.7.Assignment/Surgical/output/output.png)
<br/>

---

## Weather Prediction 🌦️

### Project Overview
The goal of this project is to predict a city’s temperature based on historical hourly weather data recorded over several years. This dataset captures temperature on an hourly basis for each day of the year.

### Implementation Steps
1. **Data Preprocessing**:
   - Converted various dates into a “day of the year” format (e.g., May 2 = day 33, December 29 = day 365).
   - Calculated the average daily temperature based on the 24 hourly measurements available for each day.
2. **Plotting Average Temperature**: A plot was created showing the average temperature for each day of the year.
   
   <br/>
   ![Average Temperature per Day Plot](https://github.com/negarslh97/Machine-Learning/blob/main/6.7.Assignment/Weather_Forecast/output/data-plot.png)
   <br/>
   <br/>

3. **Linear Perceptron Modeling**: A linear Perceptron model was trained on the processed data.

4. **Functions**:
   - **Evaluate Function**: A function to evaluate the model’s performance.
   - **Predict Function**: A function to predict temperature for a given day of the year.

5. **Weight and Bias Saving**: The trained weights and bias values were saved as `.npy` files for further use.

### Important Plots During Training
- **Loss**: Visualizes the model’s training loss across epochs.
- **Accuracy**: Shows the accuracy achieved during training.

<br/>
<br/>
![Evaluate Plot](https://github.com/negarslh97/Machine-Learning/blob/main/6.7.Assignment/Weather_Forecast/output/acc-loss.png)
<br/>

---

> **Note**: Although the dataset is not linearly separable, a linear model was intentionally chosen to meet exercise requirements. Therefore, the results are not expected to be ideal.
