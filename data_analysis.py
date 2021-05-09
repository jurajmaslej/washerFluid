import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint
import settings


def feature_analysis(df):
    for feature in settings.features_all:
        df[feature].plot(kind='hist', bins=20)
        plt.title(feature)
        plt.grid()
        plt.savefig(os.path.join(settings.feature_analysis, feature))
        plt.figure()
    plt.close()


def correlation(df, features_list):
    # Pearson (linear) correlation
    corr_pearson = df[features_list].corr(method='pearson')
    corr_pearson = corr_pearson.drop('Target')#.drop('VolumeAvg1D').drop('VolumeAvg3D').drop('VolumeAvg7D')
    corr_target_abs_values = dict()
    for feature in settings.features_target:
        corr_target_abs_values[feature] = np.abs(corr_pearson[feature].values)
    df = pd.DataFrame({'VolumeAvg1D': corr_target_abs_values['VolumeAvg1D'],
                       'VolumeAvg3D': corr_target_abs_values['VolumeAvg3D'],
                       'VolumeAvg7D': corr_target_abs_values['VolumeAvg7D']},
                      index=corr_pearson.index)
    ax = df.plot.barh(width=0.8)
    ax.grid(True)
    plt.savefig(os.path.join(settings.correlation_analysis, f"correlation_graph_sales_volume_with_volumes.png"), bbox_inches='tight')
    plt.close()

    df = pd.DataFrame({'Target': corr_target_abs_values['Target']},
                      index=corr_pearson.index)
    ax = df.plot.barh()
    ax.grid(True)
    plt.savefig(os.path.join(settings.correlation_analysis, f"correlation_graph_Target_with_volumes.png"), bbox_inches='tight')
