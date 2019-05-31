import numpy as np
import pandas as pd
from models.model import Model
from utility.series_to_supervised import series_to_supervised
from utility.data_handler import DataHandler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy import stats
from keras.optimizers import Adam, SGD


class MLP(Model):
    def __init__(self,config):
        super(MLP, self).__init__(config)
        self.dataHandler = DataHandler(nr_bins=self.configs['n_bins'])

    def architecture(self):
        n_input = self.configs['n_input']
        n_bins = self.configs['n_bins']
        model = Sequential()
        model.add(Dense(100, activation='tanh', input_dim=n_input))
        model.add(Dropout(0.1))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(n_bins, activation='softmax'))
        # sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def fit(self, train, config=None):
        self.load_config(config) if config else None
        n_input = self.configs['n_input']
        n_output = self.configs['n_output']
        n_batch = self.configs['n_batch']
        n_epochs = self.configs['n_epochs']
        verbosity = self.configs['verbosity']
        debug = True if verbosity > 0 else False
        #
        self.model = self.architecture()
        train_x, train_y = series_to_supervised(train, n_in=n_input, n_out=n_output, split=True)
        self.dataHandler.update_data(train, debug)
        y_onehot, y_bin = self.dataHandler.val_to_onehot(train_y.values)
        #
        history = self.model.fit(train_x, y_onehot,
                            epochs=n_epochs, shuffle=False, validation_split=0.2, batch_size=n_batch, verbose=verbosity)
        return history

    def predict(self, test):
        from matplotlib import pyplot as plt
        history = [x for x in self.dataHandler.data]
        predictions = []
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            n_input = self.configs['n_input']
            # prepare data
            x_input = np.array(history[-n_input:]).reshape(1, n_input)
            # forecast
            yhat = self.model.predict(x_input, verbose=0)
            #median = Model.get_horizon(np.array(self.pred_2_rv(yhat[0])).reshape(-1,1), 0.5)
            # store forecast in list of predictions
            predictions.append(yhat[0])
            # add prediction to the history for the next loop
            # other possibilities: add the real observation, add prediction with some weight of previous ones and new observations, etc.
            history.append(test[i])
            # print(yhat[0])
            # history.append(self.dataHandler.bin_centers[np.argmax(yhat[0])])
        #
        bin_prediction, _ = self.dataHandler.onehot_to_val(predictions)
        predictions = np.array(predictions)
        return predictions, bin_prediction

    def pred_2_rv(self, y_hat):
        return stats.rv_histogram((y_hat, self.dataHandler.bin_edges))

    def preds_2_rv(self, y_hats):
        """
        prediction to random variable (squashes the quantile dimension!)
        :param y_hats: (samples*hours, quantiles, variable)
        :return: predictions as a scipy random variable
        """
        assert len(y_hats.shape) == 3
        y_hat_rv = np.empty((y_hats.shape[0], 0))  # (leadtime, variable)
        for k in range(y_hats.shape[2]):  # variable
            y_tmp = []
            for i in range(y_hats.shape[0]):  # leadtime
                y_tmp.append(stats.rv_histogram((y_hats[i, :, k], self.dataHandler.bin_edges))) # todo: use pred_2_rv
            y_hat_rv = np.append(y_hat_rv, np.array(y_tmp).reshape(-1, 1), axis=1)
        return y_hat_rv


if __name__ == "__main__":
    # Test with some data
    mlp_configs = {
        'n_input': 1,
        'n_output': 1,
        'n_epochs': 10,
        'n_batch': 32,
        'lr': 0.01,
        'n_bins': 10,
        'freq': 24,
        'horizon': 24,
        'verbosity': 0
    }

    df_power = pd.read_pickle("../data/df_power.pkl")
    train, test = df_power.wf1[:200], df_power.wf1[200:250]
    print("train: ", train.shape)
    print("test: ", test.shape)

    mlp = MLP(mlp_configs)
    mlp.fit(train)
    y_hat, y_hat_bin = mlp.predict(test)
    print("y_hat: ", y_hat.shape)
    print("y_hat_bin: ", y_hat_bin.shape)
