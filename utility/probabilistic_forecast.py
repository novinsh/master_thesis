import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


class MCSimulation(object):
    """ Trajectoris created as a result of monte-carlo simulation """

    def __init__(self, nr_simulations=10):
        self.trajectories = []

    def add_trajectory(self, trj):
        self.trajectories.append(trj)

    def get_trajectory(self, j):
        self.trajectories[j]


class Trajectory(object):
    """ holding all possible lead time scenarios created as a result
        of scenario forecasting.
    """
    def __init__(self, history, debug=True):
        self.leadTimeScenarios = []
        self.histories = [history]  # it's a temporary internal trajectory
        self.histories_track = []
        self.evolved_trajectory = []
        self.debug = debug

    def add_leadtime(self, lts):
        self.leadTimeScenarios.append(lts)

    def get_leadtime(self, h):
        return self.leadTimeScenarios[h]

    def generate_histories(self, n_input, n_samples=3, test=None):
        """ generates temporary histories for the network input by sampling from
            previous step prediction. It creates (n_current_histories)*n
            new histories (temporary trajectories)
        """
        # if self.leadTimeScenarios == []:
        #    return self.histories
        pred_samples_tm1 = self.get_leadtime(-1).generate_scenarios(n=n_samples)
        histories_new = []
        for history in self.histories:
            for y_tm1 in pred_samples_tm1:
                if test is not None:
                    y_tm1_v = np.append(y_tm1, test) # vector input with other exogenous variables
                    # TODO: run the scenario with all the variables set to zero and see what's the effect (basically it would be a mc simulation alone!)
                else:
                    y_tm1_v = y_tm1
                h_new = np.append(history, y_tm1_v.reshape(1, 1, -1), axis=1)  # (trajectory, timesteps, features)
                h_new = h_new[:, -n_input:, :]
                histories_new.append(h_new)
        # print("history: ", len(self.histories))
        # print("sampleS: ", len(pred_samples_tm1))
        # print("histories news: ", len(histories_new))
        histories_new = np.array(histories_new)
        idx = np.arange(histories_new.shape[0])
        idx = np.random.choice(idx, 10 if len(idx) > 10 else len(idx), replace=False)
        self.histories = list(histories_new[idx])
        if self.debug:
            self.histories_track.append(pred_samples_tm1)
        # self.evolved_trajectory.append(histories_new[idx,0,-2:,0])
        return self.histories


class LeadTimeScenarios(object):
    """ Holding different one or more scenario(s)/prediction(s) of the model
        for each lead time and can generate new alternative futures from each
        Scenario for the next time step history
    """

    def __init__(self):
        self.scenarios = np.empty((0,))  # list of Scenario object

    def get_scenarios(self):
        return self.scenarios

    def add_scenario(self, s):
        self.scenarios = np.append(self.scenarios, s)

    def generate_scenarios(self, n=3):
        """ possible point estimates of one-step ahead by sampling from current scenarios """
        samples = np.empty((0,))
        for s in self.scenarios:
            sample_normal = np.random.normal(s.mu, s.sigma, n)
            samples = np.append(samples, sample_normal)
        return samples

    def drop_randomly(self, probability=0.5):
        """ to drop some of the scenarios with uniform probability
            (to avoid exponential growth of the trajectory)
        """
        indices_to_keep = np.random.uniform(0, 1, len(self.scenarios)) < (1 - probability)
        # for debug purposes
        print("kept %i and dropped %i" %
              (np.sum(indices_to_keep), len(self.scenarios) - np.sum(indices_to_keep)))
        self.scenarios = self.scenarios[indices_to_keep]


class Scenario(object):
    """ essentially, prediction of the model from one possible input sequence """

    def __init__(self, mu, sigma):
        self.mu = mu  # mean
        self.sigma = sigma  # standard deviation


def pred_to_musigma(pred):
    """ A function to make the code look nicer """
    return pred[0].flatten(), np.sqrt(np.exp(pred[1].flatten()))


class ProbabilisticForecast():
    """ a class that holds each leadtime as a histogram random variable """

    def __init__(self, forecast=None):
        self.forecast_variables = [] if forecast is None else forecast

    def __getitem__(self, key):
        return self.forecast_variables[key]

    def add_variable(self, scenarios, debug=False):
        samples = np.empty((0,))
        for scenario in scenarios:
            rv = stats.norm.rvs(scenario.mu, scenario.sigma, 250)
            samples = np.append(samples, rv, axis=0)
        # print(samples.shape)
        bins = np.linspace(0, 1, 100)
        hist, edges = np.histogram(samples, bins=bins, density=True)
        hist = hist / np.sum(hist)
        hist_dist = stats.rv_histogram((hist, edges))
        # debug
        if debug:
            X = np.linspace(0, 1, num=300 + 1)
            plt.plot(X, hist_dist.pdf(X))
            plt.show()
        self.forecast_variables.append(hist_dist)

    def median(self):
        median = []
        for rv in self.forecast_variables:
            median.append(rv.median())
        return np.array(median)

    def mean(self):
        mean = []
        for rv in self.forecast_variables:
            mean.append(rv.mean())
        return np.array(mean)

    def std(self):
        std = []
        for rv in self.forecast_variables:
            std.append(rv.std())
        return np.array(std)

    def quantile(self, q):
        quantile = []
        for rv in self.forecast_variables:
            quantile.append(rv.ppf(q))
        return np.array(quantile)

    def interval(self, alpha):
        lo = []
        hi = []
        for rv in self.forecast_variables:
            hi.append(rv.ppf(1 - (alpha / 2)))
            lo.append(rv.ppf(alpha / 2))
        return (np.array(lo), np.array(hi))
