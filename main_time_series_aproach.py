import os

import numpy as np
import pandas as pd
from pprint import pprint

import settings
from xlsx_to_csv import csv_from_excel
import data_analysis
import data_preparation
from analysis_time_series import variable_vs_time, moving_average, decompose


def dataframes_by_tank_id():
    df = pd.read_csv(os.path.join(settings.data_folder, settings.csv_name))
    cols = df.columns.drop('Date').drop('SIDTank')
    df.loc[:, cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df[df['Target'] > 0]
    sid_tank_unique = np.unique(df['SIDTank'].values)
    df_by_tank = []
    for tank in sid_tank_unique:
        df_by_tank.append(_sort_by_date(df[df['SIDTank'] == tank]))
    return df_by_tank


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
    df_by_tank_list = dataframes_by_tank_id()
    #variable_vs_time(df_by_tank_list, variable='Target')
    #variable_vs_time(df_by_tank_list, variable='VolumeAvg3D')
    #variable_vs_time(df_by_tank_list, variable='VolumeAvg7D')
    moving_average(df_by_tank_list[0])
    decompose(df_by_tank_list[0], variable='Target')
