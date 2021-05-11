import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import metrics
from tensorflow.keras import losses
from time import time

import settings
import plotter


class Models:

    def __init__(self, train, val, test):
        target_colum = 'Target'
        self.val = val
        self.test = test
        self.raw_train = train.copy()
        train = train.drop(columns=['Date', 'SIDTank'])
        target = train[target_colum].to_numpy()
        target_test = test[target_colum].to_numpy()
        self.target = np.array([[y, ] for y in target])
        self.target_test = np.array([[y, ] for y in target_test])
        self.target.reshape(len(self.target), 1)
        self.target_test.reshape(len(self.target_test), 1)
        self.train = train.drop(columns=[target_colum])

    def _evaluate_mae(self, prediction, target, name):
        plotter.prediction_vs_target(prediction, target, name=name)
        mae = mean_absolute_error(prediction, self.target)
        avg_target = np.mean(target)
        percentage_off = (mae / avg_target) * 100
        print(f"each prediction is {percentage_off}% off from avg target")

    def linear_regression(self):
        for feature in settings.features_high_correlation:
            x = self.train[feature].to_numpy()
            x = [[xx, ] for xx in x]
            reg = LinearRegression().fit(x, self.target)
            score = reg.score(x, self.target)
            print(f"feature {feature} score: {score}")

    def un_normalize(self, prediction, scaler, raw_data):
        train_df = raw_data.copy()
        train_df = train_df[settings.features_numeric]
        train_df['Target'] = prediction
        print(f"shape for inv transf2 {train_df.shape}")

        train_df[train_df.columns] = scaler.inverse_transform(train_df)
        return train_df['Target'].values

    def unscale_results(self, prediction, target, scaler, raw_data):
        pred_unscaled = self.un_normalize(prediction, scaler, raw_data=raw_data)
        target_unscaled = self.un_normalize(target, scaler, raw_data=raw_data)
        return pred_unscaled, target_unscaled

    def polynomial_regression(self, scaler):
        using_features = settings.features_highest_pca_mini
        poly = PolynomialFeatures(degree=len(using_features))
        x = poly.fit_transform(self.train[using_features])
        x_test = self.test.drop(columns=['Target'])
        x_test = poly.fit_transform(x_test[using_features])
        reg = LinearRegression().fit(x, self.target)
        score = reg.score(x, self.target)
        prediction = reg.predict(x)
        prediction_test = reg.predict(x_test)
        print(f"prediction test {np.where(prediction_test>100)}")
        prediction_test = np.array([[0.0000001,] if abs(i[0]) > 100 else i for i in prediction_test])
        prediction_unscaled, target_unscaled = self.unscale_results(prediction,
                                                                    self.raw_train['Target'],
                                                                    scaler, raw_data=self.raw_train)

        prediction_test_unscaled, target_test_unscaled = self.unscale_results(prediction_test,
                                                                               self.test['Target'].values, scaler,
                                                                               raw_data=self.test)

        plotter.prediction_vs_target(prediction, self.target, name="xxxxpolynom_regression_high_pca")
        plotter.prediction_vs_target(prediction_unscaled, target_unscaled, name="xxxxpolynom_regression_unscaled_high_pca")
        plotter.prediction_vs_target(prediction_test,  self.test['Target'].values, name="test_polynom_regression_high_pca")
        plotter.prediction_vs_target(prediction_test_unscaled, target_test_unscaled, name="test_polynom_regression_unscaled_high_pca")
        mae = mean_absolute_error(prediction, self.target)
        mae_unscaled = mean_absolute_error(prediction_unscaled, target_unscaled)
        mae_test = mean_absolute_error(prediction_test, self.target_test)
        mae_test_unscaled = mean_absolute_error(prediction_test_unscaled, target_test_unscaled)
        avg_target = np.mean(self.target)
        avg_target_test = np.mean(self.target_test)
        percentage_off = (mae/avg_target)*100
        percentage_off_test = (mae_test/avg_target_test)*100
        with open('polynomial_regr_scores.txt', 'a') as file:
            file.write(f"features: {using_features} score: {score},"
                       f" mae: {mae}, mae_unscaled: {mae_unscaled}, % off from avg target {percentage_off}\n"
                       f" test mae: {mae_test}, test mae unscaled {mae_test_unscaled} % off from avg target {percentage_off_test}\n")

    def _baseline_model(self, input_dim):
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(6, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss=losses.MeanSquaredError(), optimizer='adam',
                      metrics=[losses.MeanSquaredError(),
                               losses.MeanAbsoluteError(),
                               losses.MeanAbsolutePercentageError()])
        tensorboard = TensorBoard(
            log_dir=f"./logs/target_prediction_on_high_pca_percentage_loss/{str(round(time()))}",
            histogram_freq=5
        )
        keras_callbacks = [
            tensorboard
        ]
        return model, keras_callbacks

    def _evaluate_neural_net_model(self, model, keras_callback, x, y):
        #estimator = KerasRegressor(build_fn=model, epochs=10, batch_size=5, verbose=0)
        #kfold = KFold(n_splits=10)
        model.fit(x, y, epochs=20, batch_size=15, callbacks=keras_callback)
        metrics = model.evaluate(x, y)
        print(f"metrics : {metrics}")
        predictions = model.predict(x)
        print(f"predictions: {predictions}")
        print(f"target: {y}")
        mae = mean_absolute_error(predictions, self.target)
        avg_target = np.mean(self.target)
        percentage_off = (mae / avg_target) * 100
        #percentage_off_metric = losses.MeanSquaredError(predictions, self.target)
        #print(f"mae: {mae}, % off target {percentage_off}, % off metric {percentage_off_metric}")
        plotter.prediction_vs_target(predictions, y, name="neural_net_sorted_high_pca")
        #print('Accuracy: %.2f' % (accuracy * 100))
        #results = cross_val_score(estimator,x, y, cv=kfold)
        #print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

    def neural_network(self):
        using_features = settings.features_highest_pca
        x = self.train[using_features].to_numpy()
        print(f"x shape {x.shape}")
        y = np.asarray([y[0] for y in self.target])
        print(f"shape train {self.train.shape}")
        model, keras_callback = self._baseline_model(input_dim=x.shape[1])
        self._evaluate_neural_net_model(model, keras_callback, x, y)

    def svm(self, scaler):
        svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=6, epsilon=.005,
                       coef0=1)
        x = self.train[settings.features_high_correlation]
        y_true = self.target.reshape(len(self.target),1)
        prediction = svr_poly.fit(x, y_true).predict(x)
        prediction_unscaled = self.un_normalize(prediction, scaler)
        target_unscaled = self.un_normalize(self.raw_train['Target'], scaler)
        mae = mean_absolute_error(y_true, prediction)
        mae_unscaled = mean_absolute_error(prediction_unscaled, target_unscaled)
        print(f"svm MAE: {mae}, mae unscaled {mae_unscaled}")
        self._evaluate_mae(prediction, y_true, name="svm1")
