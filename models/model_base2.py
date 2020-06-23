

class ModelBase:
    def __init__(self, name, n_input, n_features, batch_size=32, verbose=1):
        self.name = name
        self.n_input = n_input
        self.n_features = n_features
        self.verbose = verbose
        self.batch_size = batch_size
        self.history = None
        self.probabilisticForecast = None

    def build_model(self, n_input, n_features, verbose=1):
        return

    def fit(self, X, y, n_epochs=1, debug=False):
        return

    def forecast(warmup, test, horizon):
        return
