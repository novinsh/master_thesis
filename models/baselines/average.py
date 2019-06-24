import numpy as np


def drift(series, horizon):
    return np.array(series[-1] + [h * ((series[-1] - series[0]) / len(series)) for h in range(horizon)])


def average(series, horizon):
    return np.array([np.mean(series)] * horizon)
