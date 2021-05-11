import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.seasonal import seasonal_decompose

import settings_time_series


def variable_vs_time(df_list, variable='Target'):
    if not os.path.exists(os.path.join(settings_time_series.graphs, f"{variable}_vs_time")):
        os.makedirs(os.path.join(settings_time_series.graphs, f"{variable}_vs_time"))
    for df in df_list:
        plt.figure()
        plt.plot(df['Date'], df[variable], label='target')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(f"{settings_time_series.graphs}", f"{variable}_vs_time/{df.at[0,'SIDTank']}.png"))
        plt.close()


def moving_average(df):
    since, to = 30, 100
    rolling = df['VolumeAvg1D'].rolling(window=2)
    rolling_mean = rolling.mean()
    df['Target'][since: to].plot(label='target')
    rolling_mean[since: to].plot(color='red', label='VolumeAvg1D roll mean', grid=True)
    plt.legend()
    plt.savefig(os.path.join(settings_time_series.graphs, 'moving_avg_from_VolumeAvg1D.png'))


def decompose(df, variable):
    result_mul = seasonal_decompose(df[variable], period=30, model='multiplicative', extrapolate_trend='freq')

    # Additive Decomposition
    result_add = seasonal_decompose(df[variable], period=30, model='additive', extrapolate_trend='freq')

    # Plot
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
    plt.savefig('decompost_multi_30.png')
    result_add.plot().suptitle('Additive Decompose', fontsize=22)
    plt.savefig('decompost_addi_30.png')
