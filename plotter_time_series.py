import matplotlib.pyplot as plt
import os
import random

import numpy as np

import settings_time_series


def prediction_vs_target(prediction, target, name="simple_model", path=settings_time_series.graphs):
    #target = np.squeeze(target)
    print(f"prediction {prediction[0]}")
    print(f"target {target[0]}")
    plt.figure()
    plt.grid(True)
    plt.plot(prediction, label='prediction', alpha=0.5)
    plt.plot(target, label='target', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(f"{path}", f"{name}.png"))
    plt.close()
