import os
import numpy as np
import pandas as pd

from xlsx_to_csv import csv_from_excel
import settings
import settings_time_series
from data_preparation import normalize_numeric
from sarimax import sarimax_estimator
from random_forest import series_to_supervised, walk_forward_validation


def datatype_cleaning():
    df = pd.read_csv(os.path.join('../', settings.data_folder, settings.csv_name))
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'],
                                       format="%Y-%m-%d",
                                       errors='raise')
    df['day_of_week'] = df['Date'].dt.strftime("%w")
    cols = df.columns.drop('Date').drop('SIDTank')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    return df


def split_df_by_station_locations(df):
    unique_locations = np.unique(df['SIDTank'].values)
    unique_locations = set([location[:location.find('_') - 1] for location in unique_locations])
    #unique_locations = set([location for location in unique_locations])  # run for each SIDTank
    grouped_by_location_df_list = []
    for location in unique_locations:
        location_df = df[df['SIDTank'].str.contains(location)]
        if len(location_df) > 100:
            grouped_by_location_df_list.append(location_df)
    return grouped_by_location_df_list


def sort_by_date(df):
    df_sorted = df.sort_values(["Date"], ascending=True)
    df_sorted = df_sorted.reset_index()
    return df_sorted


def group_by_date(df):
    agg_dict = {'SIDTank': 'first'}
    features = df.columns.drop('SIDTank').drop('Date')
    for feature in features:
        agg_dict[feature] = 'mean'
    df = df.groupby(['Date'], as_index=False).agg(agg_dict)
    return df


def fill_in_missing_dates(df):
    min_date = min(df['Date'])
    max_date = max(df['Date'])
    r = pd.date_range(start=min_date, end=max_date)
    print(f"len before {len(df)}")
    df = df.set_index('Date').reindex(r).fillna(0.0).rename_axis('dt').reset_index()
    print(f"len after {len(df)}")
    return df


def move_target_to_end(df):
    columns = list(df.columns.drop('Target'))
    target = 'Target'
    columns.append(target)
    df = df[columns]
    return df


def for_forest(df):
    df = df.drop(columns=['Date'])
    return df


def run_sarimax(grouped_by_location_df_list):
    for location_df in grouped_by_location_df_list:
        assert 'Target' not in settings_time_series.sarimax_features
        target = location_df['Target'].values
        exog = location_df[settings_time_series.sarimax_features].values
        sarimax_estimator(target=target, exog=exog, name=f"Sarimax_dates_filled_zeroes_on_locations_groupby_date/{location_df.iloc[0]['SIDTank']}")



if __name__ == '__main__':
    if os.path.exists(os.path.join('../', settings.data_folder, settings.csv_name)) is False:
        csv_from_excel()
    df = datatype_cleaning()
    df, scaler = normalize_numeric(df)
    grouped_by_location_df_list = split_df_by_station_locations(df)
    grouped_by_location_df_list = [sort_by_date(df) for df in grouped_by_location_df_list]
    grouped_by_location_df_list = [group_by_date(df) for df in grouped_by_location_df_list]
    grouped_by_location_df_list = [fill_in_missing_dates(df) for df in grouped_by_location_df_list]
    grouped_by_location_df_list = [move_target_to_end(df) for df in grouped_by_location_df_list]
    print(f"number of locations: {len(grouped_by_location_df_list)}")
    run_sarimax(grouped_by_location_df_list)

    # data = series_to_supervised(grouped_by_location_df_list, n_in=6)
    # mae, y, yhat = walk_forward_validation(data, 12)
    # print('MAE: %.3f' % mae)
