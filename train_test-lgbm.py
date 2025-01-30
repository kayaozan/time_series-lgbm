import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm as lgb

import pickle

def get_filters(df: pd.DataFrame,
                columns: list[str]) -> list[dict]:
    """Return unique value combinations of a DataFrame.

    'columns' is the list of names of the columns
    that will be searched for unique values.
    """
    
    # Select columns and drop duplicates
    # which will leave only unique rows.
    df_unique = df[columns].drop_duplicates()

    # Loop through unique rows and add them to the list.
    filter_list = []
    for i in range(len(df_unique)):
        filter_list.append(df_unique.iloc[i].to_dict())

    return filter_list

def date_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features from the date column of a DataFrame.
    
    Features are determined by ISO Calendar where the first
    week of a year is the first week containing a Thursday.
    With that in mind,
    each date is shifted to the Thursday of that week,
    and features are gathered according to that date.

    Note: this function is written for weekly data.
    It doesn't extract days of week as a feature.
    """
    
    df_date = df.select_dtypes(include='datetime')
    
    # Return the DataFrame unchanged
    # if the number of datetime columns is zero or more than one.
    if df_date.shape[1] == 0:
        print('No column of datetime dtypes!')
        return df
    elif df_date.shape[1] > 1:
        print('More than one column with datetime dtypes!')
        return df
    else:
        series_date = df_date[df_date.columns.item()]
        days_of_week = series_date.dt.dayofweek

        # Timedelta of each date to the 3rd day (Thursday) of week
        delta_of_dates = pd.Series([pd.Timedelta(days=3-day_of_week)
                             for day_of_week in days_of_week])
        series_date = series_date + delta_of_dates

        df_features = df.copy()
        df_features['year'] = series_date.dt.isocalendar().year
        df_features['quarter'] = series_date.dt.quarter
        df_features['month'] = series_date.dt.month
        df_features['week'] = series_date.dt.isocalendar().week

    return df_features

def add_lag_and_window(df: pd.DataFrame,
                       target_column: str,
                       lags: int = 1,
                       window_size: int = 4) -> pd.DataFrame:
    """ Add lag and mean of rolling window of a target column."""

    df_result = df.copy()

    # The values of target column are shifted by lag times and added to rows.
    for lag in range(1,lags+1):
        df_result[f'lag_{lag}'] = df_result.loc[:,target_column].shift(lag)

    # Mean of the rolling window of size 'window_size' is added to rows.
    df_result['rolling_window'] = df_result.loc[:,target_column].rolling(window_size).mean().shift()

    return df_result

# Read time series data
data = pd.read_pickle('Sales')

# Determine filtering columns and columns to exclude as feature
columns_to_exclude = ['Date','Quantity']
columns_to_filter = [col for col in data.columns.tolist()
                     if col not in columns_to_exclude]
columns_to_exclude.extend(columns_to_filter)

# Determine filters to examine data, add date features
filter_list = get_filters(data, columns_to_filter)
data = date_to_features(data)

# Split data into train and test sets
train_size = 0.8
dates = data['Date'].unique()
train_enddate = dates[int(0.8*len(dates))]
train_data = data.loc[data['Date'] < train_enddate]
test_data = data.loc[data['Date'] >= train_enddate]

# Gather test dates for plotting
test_dates = data['Date'].loc[data['Date'] >= train_enddate].unique()

models = []
# Loop through filters
for filterKeys in filter_list:
    # Filter by each filter combination
    mask = np.logical_and.reduce(
        [train_data[key] == value for key, value in filterKeys.items()])
    train_data_filtered = train_data[mask]
    
    # Add lag and window features
    train_data_filtered = add_lag_and_window(train_data_filtered, target_column='Quantity')
    
    # Split the train data into features and target
    X_train = train_data_filtered[[col for col in train_data_filtered.columns
                                    if col not in columns_to_exclude]]
    y_train = train_data_filtered['Quantity']

    # Initiate and fit the model
    model = lgb.LGBMRegressor()
    model.fit(X_train,y_train)

    # Initiate the plot
    figure, ax = plt.subplots(figsize=(10, 5))
    
    # Now filter the test set with the same filters
    mask = np.logical_and.reduce(
        [test_data[key] == value for key, value in filterKeys.items()])
    test_data_filtered = test_data[mask]

    test_data_filtered = add_lag_and_window(test_data_filtered, target_column='Quantity')
    
    X_test = test_data_filtered[[col for col in test_data_filtered.columns
                                 if col not in columns_to_exclude]]
    
    # Predict and store them in a DataFrame for plotting
    preds = model.predict(X_test)
    df_plot = pd.DataFrame({'date': test_dates,
                            'pred': preds})
    
    # Filter the whole data, send it to the plot
    mask = np.logical_and.reduce(
        [data[key] == value for key, value in filterKeys.items()])
    data[mask][['Date','Quantity']].plot(ax=ax, label='Actual', x='Date',y='Quantity')

    # Add the prediction to the plot
    df_plot.plot(ax=ax, label='Prediction', x='date', y='pred')

    # Arrange the legend and the title, and print the plot
    plt.legend(['Actual', 'Prediction'])
    plt.title(f'{'-'.join([value for value in filterKeys.values()])}')
    plt.show()

    # Save the model with its corresponding filters
    models.append([filterKeys, model])

# Save all models locally
with open('models', mode='wb') as f:
    pickle.dump(models, f)