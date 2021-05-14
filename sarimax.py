from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
# contrived dataset
from sklearn.metrics import mean_absolute_error

from plotter_time_series import prediction_vs_target

def process_data(df_by_tank):
    target = list(df_by_tank[:,-1])
    exog = list(df_by_tank[:,:-2])
    return target, exog


def sarimax1(df_by_tank):
    target, exog = process_data(df_by_tank)
    train_val_split = int(len(target) * 0.95)

    # fit model
    model = SARIMAX(target[:train_val_split], exog=exog[:train_val_split],
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0))

    model_fit = model.fit(disp=False, maxiter=500)
    # make prediction
    yhat = model_fit.predict(train_val_split, len(target), exog=exog[train_val_split-1:])
    yhat_on_train = model_fit.predict(0,5000, exog=exog[:5000])
    target_val = target[train_val_split - 1:]
    mae = mean_absolute_error(yhat, target_val)
    mae_train = mean_absolute_error(yhat_on_train, target[:5001])
    print(f"mae: {mae}, mae_train {mae_train}")
    prediction_vs_target(yhat, target_val, name='sarimax_day_of_the_week')

