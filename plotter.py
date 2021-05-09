import matplotlib.pyplot as plt
import os
import random

import settings


def prediction_vs_target(prediction, target, name, path=settings.models_regression):
    rand_indices_sample = random.sample(range(len(prediction)), 60)
    prediction = [prediction[i] for i in rand_indices_sample]
    target = [target[i] for i in rand_indices_sample]
    plt.figure()
    x_axis = list(range(0, 60))
    plt.grid(True)
    plt.scatter(x_axis, prediction, label='prediction',alpha=0.3)
    plt.scatter(x_axis, target, label='target',alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(f"{path}", f"{name}.png"))
