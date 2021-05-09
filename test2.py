import tensorflow as tf
import numpy as np

y_true = [12000, 200000, 300000., 60000000.]
y_pred = [6, 10, 15, 30]
mape = tf.keras.losses.MeanAbsolutePercentageError()
print(mape(y_true, y_pred).numpy())
