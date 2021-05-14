from math import sqrt

import numpy as np
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.callbacks import TensorBoard
from time import time

import plotter_time_series


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual=None, predicted=None):
    plotter_time_series.prediction_vs_target(np.squeeze(predicted), actual)

    scores = list()
    # calculate an RMSE score for each day
    for i in range(actual.shape[1]):
        # calculate mse
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        # calculate rmse
        rmse = sqrt(mse)
        # store
        scores.append(rmse)
    # calculate overall RMSE
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


# summarize scores
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


# train the model
def build_model(train_x, train_y, val_x, val_y ,n_input):
    # define parameters
    verbose, epochs, batch_size = 1, 200, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into [samples, timesteps, features]
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    print(f"n timesteps {n_timesteps}, n_features {n_features}")
    model = Sequential()
    model.add(LSTM(4, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam')


    # model = Sequential()
    # model.add(LSTM(20, activation='relu', input_shape=(n_timesteps, n_features)))
    # model.add(RepeatVector(n_outputs))
    # model.add(LSTM(20, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(100, activation='relu')))
    # model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam', metrics='mae')
    tensorboard = TensorBoard(
        log_dir=f"./logs/time_series1/{str(round(time()))}",
        histogram_freq=5
    )
    keras_callbacks = [
        tensorboard
    ]
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
              verbose=verbose, callbacks=keras_callbacks,
              validation_data=(val_x, val_y),
              )
    return model


# make a forecast
def forecast(model, history, n_input):
    # flatten data
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    # retrieve last observations for input data
    input_x = data[-n_input:, :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # forecast the next week
    yhat = model.predict(input_x, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat


# evaluate a single model
def evaluate_model(train_x, train_y, test_x, test_y, n_input):
    # fit model
    validation_size = 200
    model = build_model(train_x, train_y, test_x[:validation_size], test_y[:validation_size], n_input)
    # history is a list of weekly data
    history = [x for x in train_x]
    # walk-forward validation over each week
    predictions = list()
    print(f"lenght of test {len(test_x)}")
    for i in range(len(test_x)):
        if i % 50 == 0:
            print(f"i: {i}")
        # predict the week
        yhat_sequence = forecast(model, history, n_input)
        # store the predictions
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting the next week
        history.append(test_x[i, :])
    # evaluate predictions days for each week
    predictions = array(predictions)
    score, scores = evaluate_forecasts(actual=test_y, predicted=predictions)
    return score, scores, model, predictions


from numpy import isnan
from pandas import read_csv


# fill missing values with a value at the same time one day ago
def fill_missing(values):
    one_day = 60 * 24
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            if isnan(values[row, col]):
                values[row, col] = values[row - one_day, col]
