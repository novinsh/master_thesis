from abc import ABCMeta, abstractmethod
from configs import model_configs
from scipy import stats
import numpy as np


class Model(metaclass=ABCMeta):
    def __init__(self, config):
        self.load_config(config)

    def load_config(self, config=None):
        self.configs = model_configs
        if config is not None:
            for key, val in config.items():
                if key not in model_configs.keys():
                    raise AssertionError("Key not defined in global configurations!")
                self.configs[key] = val

    @abstractmethod
    def architecture(self):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, train, config):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, test, fh=None):
        raise NotImplementedError()

    @staticmethod
    def get_horizon(y_hat_rv, q):
        """
        Get the whole horizon for a specific quantile
        :param y_hat_rv: predictions as scipy random variables
        :param q: target quantile
        :return: prediction for all the variables for the given quantile
        """
        assert len(y_hat_rv.shape) == 2
        y_hat_q = np.empty((y_hat_rv.shape[0], 0)) # leadtime, variables
        for k in range(y_hat_rv.shape[1]): # variables
            tmp = []
            for y in y_hat_rv[:, k]: # lead time
                tmp.append(y.ppf(q))
            y_hat_q = np.append(y_hat_q, np.array(tmp).reshape(-1, 1), axis=1)
        return y_hat_q


