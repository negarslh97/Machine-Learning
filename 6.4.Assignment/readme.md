# Data Visualization and Analysis Scripts

This repository contains scripts that visualize and analyze datasets related to oysters, student performance, and the Boston housing market. Each script generates various plots to illustrate the relationships between different variables.

## Table of Contents
- [Oyster](#oyster)
- [Students_Performance](#students_performance)
- [Boston](#boston)
- [Installation](#installation)
- [Usage](#usage)

## Oyster

### Description
This script visualizes the relationship between the length and height of oysters using the abalone dataset, which includes data on various oyster species.

### Input
- **Dataset**: Abalone dataset including oyster data.

### Output
- A linear regression line plotted on the scatter plot.
![linear regression](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Oyster/output/output.png)


## Students_Performance

### Description
This script generates a synthetic dataset representing the relationship between study hours and student grades. It visualizes the correlation and compares different linear regression techniques.

### Input
- **Dataset**: Randomly generated continuous dataset for study hours (X) and student grades (Y).

### Output
1. A graph displaying hours of study versus student grades.
![hours of study versus student grades](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Students_Performance/output/output1.png)
2. Two graphs comparing the slope of the line in the self-made Linear Least Squares (LLS) algorithm and the `linregress` algorithm.
![LLS algorithm](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Students_Performance/output/output2.png)
![linregress algorithm](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Students_Performance/output/output3.png)
## Boston

### Description
This script analyzes the Boston housing dataset to visualize the relationship between housing features and prices using a 3D plot.

### Input
- **Dataset**: Boston housing dataset from `scikit-learn`.

### Output
- Correlation Chart (heatmap)
![Correlation Chart](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Boston/output/output1.png)

- A 3D scatter plot showing the actual data, along with a regression plane generated using linear regression.
![3D scatter plot](https://github.com/negarslh97/Machine-Learning/blob/main/6.4.Assignment/Boston/output/output2.png)

## Installation

To run these scripts, you need to have Python installed along with the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `scipy`

## Usage

1. Clone this repository to your local machine.
2. Navigate to the directory containing the scripts.
3. Run the desired script using Python.
