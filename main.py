import os

import numpy as np
import pandas as pd
from pprint import pprint

import settings
from xlsx_to_csv import csv_from_excel
import data_analysis
import data_preparation
from models import Models


def create_dataframe():
    df = pd.read_csv(os.path.join(settings.data_folder, settings.csv_name))
    '''
    convert all columns except 'Date' and 'SIDTank' to numeric
    '''
    df = _sort_by_date(df)
    cols = df.columns.drop('Date').drop('SIDTank')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    df = df.dropna()
    df = df[df['Target'] > 0]
    return df


def dataframe_by_tank_id():
    df = pd.read_csv(os.path.join(settings.data_folder, settings.csv_name))
    cols = df.columns.drop('Date').drop('SIDTank')
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df = df[df['Target'] > 0]
    sid_tank_unique = np.unique(df['SIDTank'].values)
    df_by_tank = []
    for tank in sid_tank_unique:
        df_by_tank.append(_sort_by_date(df[df['SIDTank'] == tank]))
    return df_by_tank


def _sort_by_date(df):
    df['Date'] = pd.to_datetime(df['Date'],
                                    format="%Y-%m-%d",
                                    errors='raise')
    df_sorted = df.sort_values(["Date"], ascending=True)
    return df_sorted


if __name__ == '__main__':
    if os.path.exists(os.path.join(settings.data_folder, settings.csv_name)) is False:
        csv_from_excel()
    df = create_dataframe()
    df_by_tank = dataframe_by_tank_id()
    df_raw = df.copy()
    df, scaler = data_preparation.normalize_numeric(df)
    data_analysis.feature_analysis(df)
    data_analysis.correlation(df, settings.features_all)

    train, validate, test = data_preparation.train_val_test_split(df, train_val_test_ratios=[0.6, 0.2, 0.2])
    #data_preparation.pca(train, validate)

    models = Models(train, validate, test)
    #models.linear_regression()
    models.polynomial_regression(scaler)

    #models.neural_network()

    #models.svm(scaler)
