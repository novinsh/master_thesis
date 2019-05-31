import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats


def naive(series, horizon):
    return np.array([series[-1]] * horizon)


def naive_quantile(series, horizon):
    # q = np.linspace(0, 1, 100)[:-1] # whatever left's for the last quantile
    zone = 1
    # sample = 1
    # arr = np.repeat(series.quantile(q).values.reshape(1, -1), horizon, axis=0)
    # label1 = [0 for i in range(arr.shape[1])]
    # label2 = [i for i in range(arr.shape[1])]
    # arr = pd.DataFrame(arr,
    #                    # index=pd.Index(test.index, names=["datetime"]),
    #                    columns=pd.MultiIndex(levels=[[1], np.round(q * 100).astype('int')], labels=[label1, label2],
    #                                          names=["wf", None]))
    # return arr
    # return np.repeat(series.quantile(q/100).values.reshape(1,-1), horizon, axis=0).reshape(sample,horizon,len(q),zone)
    q = np.linspace(0, 1, 100)
    zone = 1
    rv = stats.rv_histogram(np.histogram(series, q, density=True))
    return np.array([rv]*horizon).reshape(-1, zone)


def seasonal_naive(series, freq, horizon):
    k = np.ceil((horizon - 1) / freq).astype('int')
    return np.array([series[-k * freq + h] for h in range(horizon)])
    # return [series[-season + (i % season)] for i in range(horizon)]


def drift(series, horizon):
    return np.array(series[-1] + [h * ((series[-1] - series[0]) / len(series)) for h in range(horizon)])


def average(series, horizon):
    return np.array([np.mean(series)] * horizon)


def naive_plus(series, horizon):
    rho = acf(series, nlags=horizon + 1)
    return np.array([rho[h] * series[-1] + (1 - rho[h]) * np.mean(series) for h in range(horizon)])

