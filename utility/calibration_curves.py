import numpy as np
from scipy import stats
from matplotlib import pyplot as plt


def plot_calibration_curve_withax(rv, rv_horizon, nr_quantiles, ax, name):
    freqs = []
    quantiles = np.linspace(0, 1, nr_quantiles)
    for i in range(len(quantiles)):
        if i == 0:
            freqs.append(0)
            continue
        yy = rv.ppf(quantiles[i]) - rv.ppf(quantiles[i - 1])
        ff = rv_horizon.ppf(quantiles[i]) - rv_horizon.ppf(quantiles[i - 1])
        freqs.append(ff-yy)
    #ax.plot(quantiles, quantiles, label='Perfect Calibration', color='k')

    ax.plot(quantiles, quantiles + freqs, label=name, linewidth=2)
    #plt.plot(quantiles, np.cumsum(np.abs(freqs)), 'r-', label='integral of absolute miscalibration')
    #ax.title('Calibration Curve / %s' % np.sum(freqs))
    #ax.xlabel('Probability')
    #ax.ylabel('Frequency')
    #ax.grid()
    #ax.legend()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    #plt.show()


def plot_calibration_curve_multiple_measurements_pointestimates(ypreds_pointestimates, yreal_dev, nr_quantiles=21, ax=None, name=''):
    quantiles = []
    for i in range(len(ypreds_pointestimates)):
        yp_horizon = ypreds_pointestimates[i] # 48 * rv_historgram
        yr_horizon = yreal_dev[i]
        for j in range(len(yr_horizon)):
            yr = yr_horizon[j] # real value
            samples = np.append(yp_horizon, yr)
            samples = np.sort(samples)
            yr_idx = np.where(yr == samples)[0]
            yr_idx = yr_idx/ (len(samples)-1)
            quantiles.append(yr_idx[0])
    #plt.hist(quantiles)
    #plt.show()
    X = np.linspace(0,1,nr_quantiles)
    y=[]
    for i in range(len(X)):
        y.append(np.sum([1 for q in quantiles if q <= X[i]]))
    y = np.array(y)
    y = y / len(quantiles)
    ax.plot(np.append(0, X), np.append(0,y), label=name, linewidth=2)
    # ax.plot(X, X, 'k-')
    #plt.title('Calibration Curve %s' % name)
    ax.set_xlabel('Quantiles')
    ax.set_ylabel('Frequency')
    ax.set_xlim([-0.005,1.002])
    ax.set_ylim([-0.002,1.002])
    ax.legend()
    #plt.show()


def plot_calibration_curve_multiple_measurements_withax(ypreds_probabilistic, yreal_dev, nr_quantiles=100, debug=False, ax=None, name=''):
    quantiles = []
    for i in range(len(ypreds_probabilistic)):
        yp_horizon = ypreds_probabilistic[i] # 48 * rv_historgram
        yr_horizon = yreal_dev[i]
        for j in range(len(yr_horizon)):
            yp = yp_horizon[j]  # rv
            yr = yr_horizon[j] # real value
            samples = stats.norm.rvs(yp.mean(), yp.std(), 1000, random_state=123) # TODO: correct the sampling
            samples = np.append(samples, yr)
            samples = np.sort(samples)
            yr_idx = np.where(yr == samples)[0]
            yr_idx = yr_idx/ (len(samples)-1)
            quantiles.append(yr_idx[0])
    #plt.hist(quantiles)
    #plt.show()
    X = np.linspace(0,1,nr_quantiles)
    y=[]
    for i in range(len(X)):
        y.append(np.sum([1 for q in quantiles if q <= X[i]]))
    y = np.array(y)
    y = y / len(quantiles)
    ax.plot(np.append(0, X), np.append(0,y), label=name, linewidth=2)
    #ax.plot(X, X, 'k-')
    #plt.title('Calibration Curve %s' % name)
    ax.set_xlabel('Quantiles')
    ax.set_ylabel('Frequency')
    ax.set_xlim([-0.005,1.002])
    ax.set_ylim([-0.002,1.002])
    ax.legend()
    #plt.show()
    return

    # real data distribution
    data = []
    for y in yreal_dev:
        data.append(list(y))
    rv = stats.rv_histogram(np.histogram(data, bins=10))
    X = np.linspace(0, 1, 101)
    horizon_distribution = []
    # horizon forecast distribution
    for probabilisticForecast in ypreds_probabilistic:
        for rv in probabilisticForecast.forecast_variables:
            samples = stats.norm.rvs(rv.mean(), rv.std(), 1000, random_state=123)
            horizon_distribution.append(samples)
    rv_horizon = stats.rv_histogram(np.histogram(horizon_distribution, bins=50))
    # calibration curve
    plot_calibration_curve_withax(rv, rv_horizon, nr_quantiles, ax, name)


def plot_calibration_curve(rv, rv_horizon, nr_quantiles):
    freqs = []
    quantiles = np.linspace(0, 1, nr_quantiles)
    for i in range(len(quantiles)):
        if i == 0:
            freqs.append(0)
            continue
        yy = rv.ppf(quantiles[i]) - rv.ppf(quantiles[i - 1])
        ff = rv_horizon.ppf(quantiles[i]) - rv_horizon.ppf(quantiles[i - 1])
        freqs.append(ff-yy)
    plt.figure(figsize=(5, 5))
    plt.plot(quantiles, quantiles, label='Perfect Calibration', color='k')
    plt.plot(quantiles, quantiles + freqs, 'b.-', label='Forecast')
    #plt.plot(quantiles, np.cumsum(np.abs(freqs)), 'r-', label='integral of absolute miscalibration')
    plt.title('Calibration Curve / %s' % np.sum(freqs))
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    plt.show()


def plot_calibration_curve_multiple_measurements(ypreds_probabilistic, yreal_dev, nr_quantiles=21, debug=False):
    quantiles = []
    for i in range(len(ypreds_probabilistic)):
        yp_horizon = ypreds_probabilistic[i] # 48 * rv_historgram
        yr_horizon = yreal_dev[i]
        for j in range(len(yr_horizon)):
            yp = yp_horizon[j]  # rv
            yr = yr_horizon[j] # real value
            samples = stats.norm.rvs(yp.mean(), yp.std(), 1000, random_state=123) # TODO: correct the sampling
            samples = np.append(samples, yr)
            samples = np.sort(samples)
            yr_idx = np.where(yr == samples)[0]
            yr_idx = yr_idx/ (len(samples)-1)
            quantiles.append(yr_idx[0])
    #plt.hist(quantiles)
    #plt.show()
    X = np.linspace(0,1,100)
    y=[]
    for i in range(len(X)):
        y.append(np.sum([1 for q in quantiles if q <= X[i]]))
    y = np.array(y)
    y = y / len(quantiles)
    plt.plot(np.append(0, X), np.append(0,y))
    plt.plot(X, X, 'k-')
    plt.title('Calibration Curve')
    plt.xlabel('Quantiles')
    plt.ylabel('Frequency')
    plt.xlim([-0.0025,1.001])
    plt.ylim([-0.002,1.001])
    plt.show()
    return

    # real data distribution
    data = []
    for y in yreal_dev:
        data.append(list(y))
    rv = stats.rv_histogram(np.histogram(data, bins=10))
    X = np.linspace(0, 1, 101)
    if debug:
        plt.plot(X, rv.cdf(X), label='Data')

    horizon_distribution = []
    # horizon forecast distribution
    for probabilisticForecast in ypreds_probabilistic: # iterating over different horizons
        for rv in probabilisticForecast.forecast_variables: # iterating over each lead time of a horizon
            samples = stats.norm.rvs(rv.mean(), rv.std(), 1000, random_state=123)
            horizon_distribution.append(samples)
    rv_horizon = stats.rv_histogram(np.histogram(horizon_distribution, bins=50))
    #
    if debug:
        plt.plot(X, rv_horizon.cdf(X), label='Forecast')
        plt.title('CDF')
        plt.xlabel('Power')
        plt.ylabel('Quantiles')
        plt.legend()
        plt.show()
    #
    # calibration curve
    plot_calibration_curve(rv, rv_horizon, nr_quantiles)


def plot_calibration_curve_single_measurement(probabilisticForecast, y, nr_quantiles=21, debug=False):
    # real data distribution
    rv = stats.rv_histogram(np.histogram(y, bins=10))
    X = np.linspace(0,1,11)
    plt.plot(X, rv.pdf(X), label='Data') if debug else None
    #
    # horizon forecast distribution
    horizon_distribution = []
    for rv in probabilisticForecast.forecast_variables:
        samples = stats.norm.rvs(rv.mean(), rv.std(), 1000, random_state=123)
        horizon_distribution.append(samples)
    rv_horizon = stats.rv_histogram(np.histogram(horizon_distribution, bins=10))
    if debug:
        plt.plot(X, rv_horizon.pdf(X), label='Forecast')
        plt.title('PDF')
        plt.xlabel('Power')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    #
    # calibration curve
    plot_calibration_curve(rv, rv_horizon, nr_quantiles)

