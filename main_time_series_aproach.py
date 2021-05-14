import os

import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import lstm_multivariate
import settings
import settings_time_series
import time_series_data_prep
from xlsx_to_csv import csv_from_excel
import data_analysis
import data_preparation
from analysis_time_series import variable_vs_time, moving_average, decompose
from lstm import LstmModel
from time_series_data_prep import to_supervised
import lstm_multivariate_custom
from sarimax import sarimax1

def dataframes_by_tank_id():
    df = pd.read_csv(os.path.join(settings.data_folder, settings.csv_name))
    df['Date'] = pd.to_datetime(df['Date'])
    df['day_of_week'] = df['Date'].dt.strftime("%w")
    cols = df.columns.drop('Date').drop('SIDTank')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    #idx = pd.period_range(min(df.Date), max(df.Date))
    #df = df.reindex(idx, fill_value=0)
    df = df[df['Target'] > 0]
    df, scaler = data_preparation.normalize_numeric(df)
    sid_tank_unique = np.unique(df['SIDTank'].values)

    one_tank_df = _sort_by_date(df[df['SIDTank'] == sid_tank_unique[0]])
    one_tank_df = one_tank_df.drop(columns=['Date', 'SIDTank', 'index'])
    one_tank_df = one_tank_df[settings.time_series_features]
    if one_tank_df.shape[0] > settings_time_series.minimal_days_of_sale:
        df_by_tank = one_tank_df[:101].values

    for tank in sid_tank_unique:
        one_tank_df = _sort_by_date(df[df['SIDTank'] == tank])
        one_tank_df = one_tank_df.drop(columns=['Date', 'SIDTank', 'index'])
        one_tank_df = one_tank_df[settings.time_series_features]
        if one_tank_df.shape[0] > settings_time_series.minimal_days_of_sale:
            #df_by_tank.append(one_tank_df[:101])
            df_by_tank = np.vstack((df_by_tank, one_tank_df[:101].values))
        #df_by_tank.append(one_tank_df.values)
    print(f"tanks count {len(df_by_tank)}")
    return df_by_tank, scaler


def _sort_by_date(df):
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'],
                                    format="%Y-%m-%d",
                                    errors='raise')
    df_sorted = df.sort_values(["Date"], ascending=True)
    df_sorted = df_sorted.reset_index()
    return df_sorted


if __name__ == '__main__':
    if os.path.exists(os.path.join(settings.data_folder, settings.csv_name)) is False:
        csv_from_excel()
    df_by_tank, scaler = dataframes_by_tank_id()
    #variable_vs_time(df_by_tank_list, variable='Target')
    #variable_vs_time(df_by_tank_list, variable='VolumeAvg3D')
    #variable_vs_time(df_by_tank_list, variable='VolumeAvg7D')
    #moving_average(df_by_tank_list[0])
    #decompose(df_by_tank_list[0], variable='Target')

    sarimax1(df_by_tank)
    exit(0)

    X, y = to_supervised(np.array([df_by_tank, ]), 7)
    y = np.reshape(y, (y.shape[0], 7, 1))
    print(f"x sshape {X.shape}")
    print(f"y sshape {y.shape}")

    n_input = 7
    train_val_split = int(X.shape[0] * 0.9)
    score, scores, model, predictions = lstm_multivariate_custom.evaluate_model(train_x=X[1000:train_val_split],
                                                                                        train_y=y[1000:train_val_split],
                                                                                        test_x=X[:1000],
                                                                                        test_y=y[:1000],
                                                                                        n_input=n_input)
    lstm_multivariate_custom.summarize_scores('lstm', score, scores)
    time_series_data_prep.compare(np.squeeze(predictions), y[train_val_split:])
    print(f"score {score}")
    print(f"scores {scores}")
