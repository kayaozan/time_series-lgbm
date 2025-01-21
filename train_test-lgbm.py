import lightgbm as lgb
import numpy as np
import pandas as pd

import pickle

import matplotlib.pyplot as plt

def get_filters(df: pd.DataFrame,
                columns: list[str]) -> list[dict]:
    
    df_unique = df[columns].drop_duplicates()

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


data = pd.read_pickle('Sales')

columns_to_exclude = ['Date','Quantity']
columns_to_filter = [col for col in data.columns.tolist()
                     if col not in columns_to_exclude]
columns_to_exclude.extend(columns_to_filter)

filter_list = get_filters(data,columns_to_filter)

data = date_to_features(data)

train_size = 0.8
dates = data['Date'].unique()
train_enddate = dates[int(0.8*len(dates))]

train_data = data.loc[data['Date'] < train_enddate]
test_data = data.loc[data['Date'] >= train_enddate]
test_dates = data['Date'].loc[data['Date'] >= train_enddate].unique()

lag = 1
window = 4

models = []
for filterKeys in filter_list:
    
    mask = np.logical_and.reduce(
        [train_data[key] == value for key, value in filterKeys.items()])
    
    train_data_filtered = train_data[mask]
    train_data_filtered[f'lag_{lag}'] = train_data_filtered.loc[:,'Quantity'].shift(lag)
    train_data_filtered[f'rolling_{window}'] = train_data_filtered.loc[:,'Quantity'].rolling(window).mean().shift()
    
    X_train = train_data_filtered[[col for col in train_data_filtered.columns
                                    if col not in columns_to_exclude]]
    y_train = train_data_filtered['Quantity']

    model = lgb.LGBMRegressor()
    model.fit(X_train,y_train)

    figure, ax = plt.subplots(figsize=(10, 5))
    
    mask = np.logical_and.reduce(
        [test_data[key] == value for key, value in filterKeys.items()])
    
    test_data_filtered = test_data[mask]
    test_data_filtered[f'lag_{lag}'] = test_data_filtered.loc[:,'Quantity'].shift(lag)
    test_data_filtered[f'rolling_{window}'] = test_data_filtered.loc[:,'Quantity'].rolling(window).mean().shift()
    
    X_test = test_data_filtered[[col for col in test_data_filtered.columns
                                 if col not in columns_to_exclude]]
    
    preds = model.predict(X_test)
    
    df_plot = pd.DataFrame({'date': test_dates,
                            'pred': preds})
    
    mask = np.logical_and.reduce(
        [data[key] == value for key, value in filterKeys.items()])
    
    data[mask][['Date','Quantity']].plot(ax=ax, label='Actual', x='Date',y='Quantity')
    df_plot.plot(ax=ax, label='Prediction', x='date', y='pred')

    plt.legend(['Actual', 'Prediction'])
    plt.title(f'{'-'.join([value for value in filterKeys.values()])}')
    plt.show()

    models.append([filterKeys, model])


with open('models', mode='wb') as f:
    pickle.dump(models,f)