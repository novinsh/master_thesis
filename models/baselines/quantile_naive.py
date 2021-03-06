from models.model_base import ModelBase
from models.baselines.simple_naives import naive_quantile
from utility.probabilistic_forecast import ProbabilisticForecast


class QuantileNaive(ModelBase):
    def __init__(self, horizon=24, forecast_with_warmup=False):
        # for the naive we need horizon at the fit and forecast time so we set initially!
        super(ModelBase, self).__init__()
        self.horizon = horizon
        self.probabilisticForecast = None
        self.warmupForecast = forecast_with_warmup

    def fit(self, train, dev, **argv):
        self.probabilisticForecast = None
        if not self.warmupForecast:
            ypred_naive = naive_quantile(train, self.horizon)
            self.probabilisticForecast = ProbabilisticForecast(ypred_naive.flatten().tolist())

    def forecast(self, warmup, test=None, **argv):
        if self.warmupForecast:
            ypred_naive = naive_quantile(warmup, self.horizon)
            self.probabilisticForecast = ProbabilisticForecast(ypred_naive.flatten().tolist())
        ypred_median = self.probabilisticForecast.median()
        return ypred_median

    def forecast_probabilistic(self, warmup, test=None, **argv):
        if self.warmupForecast:
            ypred_naive = naive_quantile(warmup, self.horizon)
            self.probabilisticForecast = ProbabilisticForecast(ypred_naive.flatten().tolist())
        return self.probabilisticForecast


class QuantileNaiveNew(ModelBase):
    """ Simplified verison of the above class, make use of this everywhere! """
    def __init__(self):
        super(ModelBase, self).__init__()
        self.probabilisticForecast = None
        self.train_data = None

    def fit(self, train, dev=None, **argv):
        self.train_data = train

    def forecast(self, warmup, test, **argv):
        ypred_naive = naive_quantile(self.train_data, horizon=len(test))
        self.probabilisticForecast = ProbabilisticForecast(ypred_naive.flatten().tolist())
        return self.probabilisticForecast.median()

    def forecast_probabilistic(self, warmup, test=None, **argv):
        ypred_naive = naive_quantile(self.train_data, horizon=len(test))
        self.probabilisticForecast = ProbabilisticForecast(ypred_naive.flatten().tolist())
        return self.probabilisticForecast
