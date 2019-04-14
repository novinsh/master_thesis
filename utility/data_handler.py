from scipy import stats
from keras.utils import np_utils
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler


class DataHandler():
    def __init__(self, data=None, nr_bins=10, debug=False):
        self.nr_bins = nr_bins
        self.data = None
        if data is not None:
            self.update_data(data, debug)

    def plot_debug(self):
        X = np.linspace(np.min(self.data), np.max(self.data), 100)
        plt.figure(figsize=(16, 5))
        plt.title("PDF and CDF")
        # plt.hist(self.data, bins=self.bin_edges, density=False, label="Histogram")
        #plt.plot(X, self.hist_dist.pdf(X), label='PDF')
        plt.plot(X, self.hist_dist.cdf(X), label='CDF')
        plt.xticks(self.bin_centers, np.round(self.bin_centers, 2), rotation=15)
        plt.vlines(self.bin_edges, 0, 1, colors="grey", lw=1, linestyles="dashed")
        plt.scatter(self.bin_centers, self.hist_dist.cdf(self.bin_centers), marker="*", label="bin center",
                    edgecolors="red", zorder=3)
        plt.ylabel("quantiles")
        plt.xlabel("x")
        plt.legend(loc='lower right')
        centerText = [str(b + 1) for b in range(self.nr_bins)]
        for i in range(self.nr_bins):
            plt.text(self.bin_centers[i], 1, centerText[i])
        plt.show()
        hist, _ = np.histogram(self.data, bins=self.bin_edges)  #
        plt.bar(range(len(hist)), hist)
        plt.xticks(range(len(hist)), range(1, len(hist) + 1))
        plt.title('Histogram with new binning')
        plt.show()

    def update_data(self, data, debug):
        self.data = data
        bins=np.linspace(0,1,self.nr_bins)
        hist, edges = np.histogram(self.data, bins=bins, density=True)
        hist /= np.sum(hist)
        self.hist_dist = stats.rv_histogram((hist, edges))
        # calculate the bin edges by dividing the quantiles equally on the CDF:
        quantiles = np.linspace(0, 1, num=self.nr_bins + 1)  # quantile edges from the right
        self.bin_edges = self.hist_dist.ppf(quantiles)  # invCDF to find the bin edges
        self.bin_centers = np.array(self.bin_edges[:-1] + np.diff(self.bin_edges) / 2)
        self.plot_debug() if debug else None

    def get_data(self):
        return self.data

    def get_variable(self):
        """ returns the histogram random variable (scipy) representing the data distribution """
        return self.hist_dist

    def closest_bin_on_cdf(self, value):
        """ assign bin on the cdf """
        if value < np.min(self.data) or value > np.max(self.data):
            print("value must range between {0} and {1}".format(np.min(self.data), np.max(self.data)))
            assert False
        dist_1 = np.abs(self.bin_centers - value)
        return np.argmin(dist_1)

    def val_to_onehot(self, val):
        """ Take a vector val timeseries and assigns a bin based on cdf or invcdf to each value in y """
        # TODO: extend for the multivariate case
        v_to_bin = []
        for v in val:
            v_to_bin.append(self.closest_bin_on_cdf(v))
        # print(len(v_to_bin))
        # print(self.nr_bins)
        return np_utils.to_categorical(v_to_bin, num_classes=self.nr_bins), v_to_bin

    def onehot_to_val(self, onehot):
        # TODO: extend for the multivariate case
        bin_num = np.argmax(onehot, axis=1)  # corresponding bin number
        y = self.bin_centers[bin_num]
        return y, bin_num

    def get_cdf(self):
        X = np.linspace(np.min(self.data), np.max(self.data), 100)
        return X, self.hist_dist.cdf(X)

    def plot_on_dist(self, x):
        # PDF and CDF
        X = np.linspace(np.min(self.data), np.max(self.data), 100)
        q = self.hist_dist.cdf(x)
        plt.plot(X, self.hist_dist.cdf(X), label="CDF $F_X(x)$")
        plt.plot(x, q, ".r", label="x on cdf")
        plt.plot(X, self.hist_dist.pdf(X), label="PDF $f_X(x)$")
        plt.plot(x, self.hist_dist.pdf(x), ".g", label="x on pdf")
        plt.xticks(self.bin_centers, np.round(self.bin_centers, 2), rotation=15)
        plt.vlines(self.bin_edges, 0, 1, colors="grey", lw=1, linestyles="dashed")
        plt.xlabel("x")
        plt.ylabel("probability")
        centerText = [str(b) for b in range(self.nr_bins)]
        for i in range(self.nr_bins):
            plt.text(self.bin_centers[i], 1, centerText[i])
        plt.legend()
        plt.show()
        # inverse of CDF
        plt.plot(self.hist_dist.cdf(X), X)
        plt.plot(q, x, ".r")
        x_obtained_from_q = self.hist_dist.ppf(q)
        plt.hlines(x_obtained_from_q, 0, q, colors='r', linestyles='dashed')
        plt.vlines(np.arange(0, 1 + 1 / self.nr_bins, 1 / self.nr_bins),
                   np.min(X), np.max(X), colors="grey", lw=1, linestyles="dashed")
        centerText = ["Q" + str(b + 1) for b in range(self.nr_bins)]
        plt.scatter(self.bin_centers_invcdf, self.hist_dist.ppf(self.bin_centers_invcdf), marker="*",
                    label="bin center", edgecolors="red", zorder=3)
        # plt.xticks(self.bin_edges_invcdf, np.round(self.bin_edges_invcdf), rotation=15)
        for i in range(self.nr_bins):
            plt.text(self.bin_centers_invcdf[i] - 0.024, np.max(X), centerText[i])
        plt.title("inverse CDF $x = F^{-1}(Q)$")
        plt.xlabel("Quantile")
        plt.ylabel("X")
        plt.show()


if __name__ == "__main__":
    import timesynth as ts

    # Initializing TimeSampler
    time_sampler = ts.TimeSampler(stop_time=500)
    # Sampling irregular time samples
    irregular_time_samples = time_sampler.sample_irregular_time(num_points=10000, keep_percentage=20)
    # Initializing Sinusoidal signal
    sinusoid = ts.signals.Sinusoidal(frequency=0.1)
    # Initializing Gaussian noise
    white_noise = ts.noise.GaussianNoise(std=0.1)
    # Initializing TimeSeries class with the signal and noise objects
    timeseries = ts.TimeSeries(sinusoid, noise_generator=white_noise)
    # Sampling using the irregular time samples
    samples, signals, errors = timeseries.sample(irregular_time_samples)

    dataHandler = DataHandler(samples, nr_bins=10, debug=True)