import numpy as np
from sklearn.metrics import mean_absolute_error


def to_supervised(train, n_input, n_out=7):
    # flatten data
    train = np.array(train)

    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end <= len(data):
            # delete target from train data
            x_without_target = np.delete(data[in_start:in_end, :], -1, axis=1)
            X.append(x_without_target)
            y.append(data[in_end:out_end, -1])  # -1th column is Target
        # move along one time step
        in_start += 1
    return np.array(X), np.array(y)


def compare(predictions, target):
    mae_avg = []
    mae_predicted_avg = []
    for week_prediction, week_target in zip(predictions, target):
        mae = mean_absolute_error(week_prediction, week_target)
        mae_predicted_avg.append(np.mean(week_prediction))
        mae_avg.append(mae)
    print(f"avg mae was {np.mean(mae_avg)}, avg prediction {np.mean(mae_predicted_avg)}")
