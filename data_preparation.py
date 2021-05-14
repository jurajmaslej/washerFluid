from pprint import pprint
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import settings


def normalize_numeric(df):
    #plt.plot(df['Target'].values, ':')
    #plt.savefig('target.png')
    df_numeric = df[settings.features_numeric]
    scaler = MinMaxScaler()
    df_numeric[df_numeric.columns] = scaler.fit_transform(df_numeric)
    for column in df_numeric.columns:
        df[column] = df_numeric[column]
    return df, scaler


def train_val_test_split(df, train_val_test_ratios=None):
    if train_val_test_ratios is None:
        train_val_test_ratios = [0.6, 0.2, 0.2]
    train, validate, test = np.split(df.sample(frac=1, random_state=42),
                                     [int(train_val_test_ratios[0] * len(df)),
                                      int((train_val_test_ratios[0] + train_val_test_ratios[1]) * len(df))])
    #print(f"shape train {train.shape}")
    #print(f"shape val {validate.shape}")
    #print(f"shape test {test.shape}")
    return train, validate, test


def pca(train, validate):
    train_numeric = train[settings.features_numeric]
    X_train = train_numeric.drop(columns=['Target'])
    validate_numeric = validate[settings.features_numeric]
    X_val = validate_numeric.drop(columns=['Target'])
    pca = PCA()
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    features_num_X = settings.features_numeric.copy()
    features_num_X.remove('Target')
    explained_variance = {key: [variance, ] for key, variance in zip(features_num_X, pca.explained_variance_ratio_)}
    explained_variance = pd.DataFrame.from_dict(explained_variance, orient='index', columns=['pca_variance'])
    explained_variance.sort_values(["pca_variance"], ascending=False, inplace=True)

    ax = explained_variance.plot.barh(width=0.8)
    ax.grid(True)
    plt.savefig(os.path.join(f"{settings.feature_analysis}", f"pca.png"), bbox_inches='tight')
    pprint(explained_variance)
