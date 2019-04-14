import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats

from utility.evaluation import smape, mase, mae, nae


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
    rv = stats.rv_histogram(np.histogram(series.values, q, density=True))
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


if __name__ == "__main__":
    df_power = pd.read_pickle("data/df_power.pkl")

    horizon = 48
    freq = 24

    train = df_power.loc['2012-01-01':'2012-01-14 00:00:00'].wf1
    test = df_power.loc['2012-01-14 01:00:00':'2012-01-16 00:00:00'].wf1 # two days

    def plot_result(x, y, y_pred, title=""):
        offset = 2 * horizon  # how much of training data to plot
        plt.plot(range(offset), x[-offset:])
        plt.plot(range(offset, offset + horizon), y)
        plt.plot(range(offset, offset + horizon), y_pred)
        plt.title(title)
        plt.show()


    def evaluation(y_test, y_pred):
        e1 = smape(y_test.values, y_pred)
        e2 = mase(train, y_test.values, y_pred, freq)
        e3, _, _ = mae(nae(y_test.values, y_pred, period=horizon, quantiles=1, zones=1))
        print("smape: {0:.3f}".format(e1))
        print("mase: {0:.3f}".format(e2))
        print("mae: {0:.3f}".format(e3))


    print("### naive")
    y_pred = naive(train, horizon)
    plot_result(train, test, y_pred, title="naive")
    evaluation(test, y_pred)

    print("### seasonal naive")
    y_pred = seasonal_naive(train, freq, horizon)
    plot_result(train, test, y_pred, title="seasonal naive")
    evaluation(test, y_pred)

    print("### drift")
    y_pred = drift(train, horizon)
    plot_result(train, test, y_pred, title="drift")
    evaluation(test, y_pred)

    print("### average")
    y_pred = average(train, horizon)
    plot_result(train, test, y_pred, title="average")
    evaluation(test, y_pred)

    print("### improved naive")
    y_pred = naive_plus(train, horizon)
    plot_result(train, test, y_pred, title="improved naive")
    evaluation(test, y_pred)