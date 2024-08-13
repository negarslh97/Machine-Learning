# Assignment 6.1

## KNN Method
### Description

Here we define the KNN (K Nearest Neighbors) method.
We made a dataset of the width and length of the fruits
and we trained our model on it and then we tested it on another dataset.

### Diagram

<b>Diagram of width and length of fruits in main code</b>

<br>

![Chart of apples and bananas and watermelons](https://github.com/negarslh97/Machine-Learning/blob/main/6.1.Assignment/KNN/output/output.png)

<br/>
<br/>

<b>Diagram of width and length of fruits in test code</b>

<br>

![Chart of apples and bananas and watermelons](https://github.com/negarslh97/Machine-Learning/blob/main/6.1.Assignment/KNN/output/output-test.png)

<br/>
<br/>

### Code

<b>We used the following function to create the dataset</b>

```python
np.random.normal()
```

<b>We used Euclidean distance to calculate the distance between points</b>

Euclidean distance function :

```python
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
```