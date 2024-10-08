# Tehran House Price Prediction

This notebook explores the prediction of house prices in Tehran using two different regression methods: Linear Least Squares (LLS) and Linear Regression.

## Description

The dataset contains information about house prices in Tehran, focusing on the most expensive properties. The goal of the project is to predict house prices using different machine learning models and evaluate their performance based on the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### The top 5 most expensive houses

| Address        | Price(Toman)              |
|----------------|--------------------|
| Zaferanieh| 924,000,000,000    |
| Abazar    | 910,000,000,000    |
| Lavasan    | 850,000,000,000    |
| Ekhtiarieh | 816,000,000,000    |
| Niavaran  | 805,000,000,000    |
<br/>
<br/>

![5 expensive houses](https://github.com/negarslh97/Machine-Learning/blob/main/6.5.Assignment/Tehran_House_Price/output/expemsive.png)

<br/>
<br/>

### Train and Test
To choose the best training and testing data, we must check the correlation
<br/>
<br/>

![correlation](https://github.com/negarslh97/Machine-Learning/blob/main/6.5.Assignment/Tehran_House_Price/output/correlation.png)

<br/>
<br/>

### Evaluation Metrics
The following metrics were used to evaluate the models:
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error

### Results

The performance comparison between the two models is as follows:

| Metric  | LLS (Linear Least Squares) | Linear Regression      | Sklearn RidgeCV |
|---------|----------------------------|------------------------|------------------------|
| MAE     | 3,089,735,246.55           | 2,928,317,487.84       | 2,927,934,644.35       |
| MSE     | 3.7002187134535e+19        | 3.3529333701534188e+19 | 3.3528996743783084e+19 |
| RMSE    | 6,082,942,308.99           | 5,790,451,942.77       | 5790422846.717075      |

### Conclusion

LLS and Linear Regression perform similarly well, with Linear Regression having a slight edge when it comes to MSE and RMSE. However, RidgeCV does better in terms of RMSE, but it shows more error in MAE and MSE, which might suggest that its predictions are a bit off balance.

# Tehran House Price Prediction
# Dollar to Rial Price Analysis

This notebook examines the changes in the Dollar to Rial exchange rate during the presidencies of Ahmadinejad, Rouhani, and Raisi. The analysis includes both the highest and lowest prices recorded during each presidency, as well as the Mean Absolute Error (MAE) of the dollar prices.

## Description

The dataset focuses on the exchange rates of the Dollar to Rial during different presidential terms in Iran. The aim is to evaluate the fluctuations in dollar prices and compare the MAE for each presidency.

### Data

| President     | Highest Dollar Price (Rial) | Lowest Dollar Price (Rial) | MAE        |
|---------------|-----------------------------|----------------------------|------------|
| Ahmadinejad   | 39,700                       | 13,350                      | 48.4998    |
| Rouhani       | 99,950                       | 100,370                     | 475.1518   |
| Raisi         | 555,600                      | 206,210                     | 1279.6019  |

### MAE loss
- MAE of ahmadinejad is 2.0110322469723354e+24.
- MAE of rohani is 1.6516357696253647e+26.
- MAE of raisi is 1.498722684643541e+27.

### Results

- During Ahmadinejad's presidency, the dollar prices fluctuated between 13,350 and 39,700 Rials, with an MAE of 48.50.
- During Rouhani's presidency, the dollar prices fluctuated between 100,370 and 99,950 Rials, with an MAE of 475.15.
- During Raisi's presidency, the dollar prices fluctuated significantly between 206,210 and 555,600 Rials, with the highest MAE of 1279.60.

### Conclusion

The exchange rate fluctuations have been most significant during Raisi's presidency, with a notable increase in the highest and lowest prices compared to the previous presidencies. The MAE has also been the highest, reflecting greater unpredictability in the exchange rate during this period.