from scipy.interpolate import interp1d
import numpy as np
from models.model import Model


def smape(y_test, y_pred):
    """
    Calculates sMAPE

    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    """
    a = np.reshape(y_test, (-1,))
    b = np.reshape(y_pred, (-1,))
    return np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()


def mase(insample, y_test, y_hat_test, freq):
    """
    Calculates MAsE

    :param insample: insample data
    :param y_test: out of sample target values
    :param y_hat_test: predicted values
    :param freq: data frequency
    :return:
    """
    y_hat_naive = []
    for i in range(freq, len(insample)):
        y_hat_naive.append(insample[(i - freq)])

    masep = np.mean(abs(insample[freq:] - y_hat_naive))
    return np.mean(abs(y_test - y_hat_test)) / masep


def nae(y_test, y_pred, period=24, quantiles=1, zones=1):
    """
    number of absolute errors

    :param y_test: the real or ground truth data
    :param y_pred: the prediction which could include (samples, period, resolution, zone)
    :
    """
    median = quantiles // 2  # pick the median for MAE calculation
    return np.abs(y_test.reshape(-1, period, zones) - y_pred.reshape(-1, period, quantiles, zones)[:, :, median, :])


def mae(error):
    """
    mean absolute error

    :param error: absolute error
    :return: tatal mean, mae over horizon, std over horizon
    """
    return np.mean(error), np.mean(error, axis=(0, 2)), np.std(error, axis=(0, 2))


def improvement(h_ref, h):
    """
    Calculating relative improvement of a model over a measure

    :param h_ref: reference measurement
    :param h: new approach's measurement
    :return: improvement relative to the baseline (ranging from -inf to +inf)
    """
    return (h_ref - h) / h_ref


def evaluate(test, y_pred, period=24, bins=99, zones=1):
    """
    number of Continuous Ranked Probability Scores (CRPSs)
    https://www.lokad.com/continuous-ranked-probability-score
    https://en.wikipedia.org/wiki/Brier_score

    :param test: time series data
    :param prediction: probabilistic distribution forecast (samples, period, bins, zones), bins could be different from the target quantiles
    :param period: number of steps for the horizon
    :param bins: number of bins in the prediction CDF
    :param zones: number of forecasted series (target variable(s))
    :return: NCRPS measure
    """
    quantiles = 100
    y = np.linspace(0, 1, 100)
    y_real = test.values.reshape(-1, period, zones)  # days, hours, zones
    # print(y_real.shape)
    # print(y_pred.shape)
    # match up the predictions CDF quantiles with the desired y quantiles
    F_hat = np.empty((len(y), y_real.shape[0], y_real.shape[1], y_real.shape[2]))
    h=-1
    for i in range(y_real.shape[0]):
        for j in range(y_real.shape[1]):
            h += 1
            for k in range(y_real.shape[2]):
                F_hat[:, i, j, k] = y_pred[h,k].cdf(y)
    #
    I = np.heaviside(np.moveaxis(y - y_real[:, :, :, np.newaxis], -1, 0), 0)
    acc = (F_hat - I) ** 2
    #
    y_pred = Model.get_horizon(y_pred,.5).reshape(y_real.shape)
    return np.trapz(acc, y, axis=0), np.abs(y_pred - y_real)



# def evaluate(test, prediction, period=24, bins=99, zones=1):
#     """
#     number of Continuous Ranked Probability Scores (CRPSs)
#     https://www.lokad.com/continuous-ranked-probability-score
#     https://en.wikipedia.org/wiki/Brier_score
#
#     :param test: time series data
#     :param prediction: probabilistic distribution forecast (samples, period, bins, zones), bins could be different from the target quantiles
#     :param period: number of steps for the horizon
#     :param bins: number of bins in the prediction CDF
#     :param zones: number of forecasted series (target variable(s))
#     :return: NCRPS measure
#     """
#     quantiles = 100
#     y = np.linspace(0, 1, 100)
#     y_real = test.reshape(-1, period, zones)  # days, hours, zones
#     print(y_real.shape)
#     y_hat = prediction.values.reshape(-1, period, bins, zones)  # days, hours, bins, zones
#     print(y_hat.shape)
#     # match up the predictions CDF quantiles with the desired y quantiles
#     F_hat = np.empty((len(y), y_hat.shape[0], y_hat.shape[1], y_hat.shape[3]))
#     for i in range(y_hat.shape[0]):
#         for j in range(y_hat.shape[1]):
#             for k in range(y_hat.shape[3]):
#                 # F = interp1d(np.append(y_hat[i,j,:,k],1), np.append(df_bench.columns.levels[1].values/100,1), kind='linear')
#                 F = interp1d(np.append(y_hat[i, j, :, k], 1), np.append(prediction.columns.levels[1].values / 100, 1),
#                              kind='linear', fill_value='extrapolate') # TODO: extrapolation sounds like not a good idea
#                 F_hat[:, i, j, k] = F(y)
#     #
#     I = np.heaviside(np.moveaxis(y - y_real[:, :, :, np.newaxis], -1, 0), 0)
#     acc = (F_hat - I) ** 2
#     #
#     return np.trapz(acc, y, axis=0), np.abs(y_hat[:, :, 50, :] - y_real)
