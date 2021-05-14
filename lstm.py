import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
import numpy as np
from keras import models, layers
from plotter_time_series import prediction_vs_target


class LstmModel:

    def __init__(self, df_list, scaler):
        self.df_list = df_list
        self.x, self.y = self.data_split()
        self.scaler = scaler

    def data_split(self):
        tanks_df = self.df_list.copy()
        X, Y = [], []
        for df in tanks_df:
            x = df.drop(['Target'], axis=1).drop(['Date'], axis=1).drop(['SIDTank'], axis=1)
            y = df['Target']
            X.append(x)
            Y.append(y)
        return X, Y

    def build_model(self, n_timesteps):
        n_samples = len(self.x)
        _, n_features = self.x[0].shape
        print(f"n timesteps {n_timesteps}, n_features {n_features}")

        # lstm_input = layers.Input(shape=(None, n_features))
        # lstm_output = layers.LSTM(n_samples)(lstm_input)
        # model = models.Model(lstm_input, lstm_output)

        model = Sequential()
        model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])
        return model

    def train_on_one_tank(self, model, train_val_split):
        x = self.x[0]
        y = self.y[0]
        x_train = np.array([x.values[:train_val_split-1], ])
        y_train = np.array([y.values[:train_val_split-1], ])
        x_val = np.array([x.values[train_val_split:-1], ])
        y_val = np.array([y.values[train_val_split:-1], ])
        model.fit(x_train, y_train, epochs=200)
        eval = model.evaluate(x_val, y_val)
        print(f"model eval on validation: {eval}")
        pred1 = model.predict(x_train)
        pred2 = model.predict(x_val)
        print(f"prediction : {pred1}, {pred2}")
        print(f"target : {y_train[-1][-1]}, {y_val[-1][-1]}")

    def train_model(self, model):
        train_val_split = int(len(self.x) * 0.8)
        xx_train = np.array([self.x[0].values[:train_val_split]],)
        yy_train = np.array([self.y[0].values[:train_val_split]],)
        xx_val = np.array([self.x[0].values[train_val_split:]], )
        yy_val = np.array([self.y[0].values[train_val_split:]], )
        for x_df, y_df in zip(self.x, self.y):
            print(f"values shape {x_df.shape}, type{type(x_df.values)}")
            xx_train = np.vstack((xx_train, np.array([x_df.values[:train_val_split], ])))
            yy_train = np.vstack((yy_train, np.array([y_df.values[:train_val_split], ])))
            xx_val = np.vstack((xx_train, np.array([x_df.values[:train_val_split], ])))
            yy_val = np.vstack((yy_train, np.array([y_df.values[:train_val_split], ])))
        model.fit(xx_train, yy_train, epochs=20)
        eval = model.evaluate(xx_train, yy_train)
        print(f"model eval on validation: {eval}")
        prediction = model.predict(xx_train)
        print(f"prediction shape {prediction.shape}")
        print(f"prediction  {prediction}")
        prediction_vs_target(prediction, yy_train)

    def run(self):
        train_val_split = int(len(self.x[0]) * 0.8)
        model = self.build_model(train_val_split-1)
        self.train_on_one_tank(model, train_val_split)
