# Time Series Forecasting with LightGBM
This repository contains sample code to train and test an LGBM Regressor which aims to forecast future time series data.

## Data Exploration
The data consists of weekly sales quantities of products with various categories, subcategories and season. Here is a sample part of the data.

| Date | Category | SubCategory | Season | Quantity |
|---|---|---|---|---:|
|  2024-01-07 | Category1 | SubCategory1 | Summer | 706 |
|  2024-01-07 | Category1 | SubCategory1 | Winter | 871 |
|  2024-01-07 | Category1 | SubCategory2 | Summer  | 123 |
|  2024-01-07 | Category1 | SubCategory2 | Winter  | 163 |

It's always a good idea to start with a general picture. The following is the data grouped and summed by date.

![whole data](https://github.com/user-attachments/assets/40e016c4-349f-4d5a-ad06-285abedbeec4)

Looks messy, isn't it? The general picture doesn't tell much, it's hard to see any patterns.

However, after filtering by category, subcategory and season, the trend and seasonality becomes clearer.

![filtered data - 1](https://github.com/user-attachments/assets/4dfd2bb4-3dd4-4c2e-a0f3-220812d774b7)

![filtered data - 2](https://github.com/user-attachments/assets/ed918455-f764-4cda-b32a-50cdf5c8ff4c)

The pictures are similar mostly for other categories as well. Sales of each category increase in its corresponding season, therefore it is beneficial to train the model by filtering the data first.

## Preparation

As mentioned, data is filtered by category, subcategory and season. Before training, more manipulations are required.

- Data is given by timestamps, and these can be split into features which model can gather information.
  
  Consequently, year, quarter, month and week of timestamps are generated and added to the data.

- Since that is a time series data, target value of each week will also be related to its previous weeks.
  
  After examination, value of the previous week and mean of the values of 4 previous weeks are also added.

Here is how the data looks right before training:

| Date | Category | SubCategory | Season | year | quarter | month | week | lag_1 | rolling_window | Quantity
|---|---|---|---|---| --- | --- | --- | ---:| ---:| ---:|
|	2022-01-02 | Category1 | SubCategory1 | Winter | 2021 | 4 | 12 | 52 | 1365 | 1419.25 | 1546
|	2022-01-09 | Category1 | SubCategory1 | Winter | 2022 | 1 | 1 | 1 | 1546 | 1484.50 | 1552
|	2022-01-16 | Category1 | SubCategory1 | Winter | 2022 | 1 | 1 | 2 | 1552 | 1525.00 | 1388
|	2022-01-23 | Category1 | SubCategory1 | Winter | 2022 | 1 | 1 | 3 | 1388 | 1462.75 | 1271
|	2022-01-30 | Category1 | SubCategory1 | Winter | 2022 | 1 | 1 | 4 | 1271 | 1439.25 | 1380

After that, we can just discard the Date column and use these features.

## Results

Data is filtered, trained and tested as demonstrated in `train_test-lgbm.py`. Some results are plotted and shown below.

In this one, the model is able to follow the descending trend of data.

![result 1](https://github.com/user-attachments/assets/1df6b996-bcae-4e51-ac1c-cb5b50c990fa)

Next one is another filtered data. While predictions are close to expected values at the beginning, the model fails to keep up with the oscillations. There's also an unseen peak right at the end which the model fails to foresee.

![result 2](https://github.com/user-attachments/assets/ff7c77be-834b-40f2-a98f-92fa18e0c55c)

This is an interesting one. Although this subgroup of product is considered as a summer product, it has many irregularities throughout the year. Overall, the model keeps a good pace with the real values.

![result 3](https://github.com/user-attachments/assets/7fe70839-3d8c-4800-b0a7-fbaee8f456fe)

## Possible Improvements

The results are mostly satisfactory, but there's always room for improvement.

- There is no hyperparameter optimization included here, all of the work is left to default values.

  Techniques such as GridSearch, or an optimization framework like Optuna can be used in the process.
  
- More features can be generated. Sales are expected to be affected by holidays, and dates coinciding with holidays can be marked in preparation.

  If possible, external data that might be related to sales can also be added, such as stock, weather etc.
