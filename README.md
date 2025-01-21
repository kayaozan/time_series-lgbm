WORK IN PROGRESS

# Time Series Forecasting with LightGBM
Sample code to train and test an LGBM Regressor which can forecast future time series data.

## Data Exploration
The data consists of weekly sales quantities of products with various categories, subcategories and season. Here is a sample part of the data.

| Date | Category | SubCategory | Season | Quantity |
|---|---|---|---|---:|
|  2024-01-07 | Category1 | SubCategory1 | Summer | 706 |
|  2024-01-07 | Category1 | SubCategory1 | Winter | 871 |
|  2024-01-07 | Category1 | SubCategory2 | Summer  | 123 |
|  2024-01-07 | Category1 | SubCategory2 | Winter  | 163 |

It's always a good idea to start with a general picture. The following is the data grouped and summed by each week.

![image](https://github.com/user-attachments/assets/40e016c4-349f-4d5a-ad06-285abedbeec4)

Looks messy, isn't it? The general picture doesn't tell much, it's hard to see any patterns. However, after filtering by category, subcategory and season, the trend and seasonality becomes clearer.

![image](https://github.com/user-attachments/assets/4dfd2bb4-3dd4-4c2e-a0f3-220812d774b7)

![image](https://github.com/user-attachments/assets/ed918455-f764-4cda-b32a-50cdf5c8ff4c)

## Results
