import keras
from models.model_base import ModelBase
from models.RNN.common import *
from keras.models import Model, Input
from keras.layers import GRU, Dense
from utility.clr_callback import *
from utility.visualization import *
from utility.probabilistic_forecast import *


class ModelEpistemic(ModelBase):
    def build_model(self, n_input, n_features):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in, training=True)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        model = Model(inputs=x_in, outputs=mean)
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model

    def fit(self, train, dev, n_input=24, debug=False):
        n_features = train.shape[1]

        generator = TimeseriesGenerator(train, train[:, 0], length=n_input)
        dev_generator = TimeseriesGenerator(dev, dev[:, 0], length=n_input)
        batch_size = 32
        epoch_steps = train.shape[0] / batch_size

        self.model_ep = self.build_model(n_input, n_features)
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)
        self.history = self.model_ep.fit_generator(generator, validation_data=dev_generator, shuffle=False,
                                                   epochs=10, verbose=1, steps_per_epoch=epoch_steps,
                                                   callbacks=[clr_cb])
        self.n_input = n_input

    def forecast(self, warmup, test, nc_simulation=10):
        n_input = self.n_input

        def neuralnet_prediction(warmup, test):
            history = warmup.reshape(1, -1, 5)
            predictions = []
            ypred = self.model_ep.predict(history)
            predictions.append(ypred[0][0])
            #
            for h in range(len(test) - 1):
                x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
                history = np.append(history, x_tp1, axis=1)
                history = history[:, -n_input:, :]
                ypred = self.model_ep.predict(history)
                predictions.append(ypred[0][0])
            #
            predictions = np.array(predictions)
            return predictions

        #
        horizon = len(test)
        mc_preds = np.empty((0, horizon))
        for i in range(nc_simulation):  # parallelize this part of the code
            ypred = neuralnet_prediction(warmup, test[:horizon])
            mc_preds = np.append(mc_preds, ypred.reshape(1, -1), axis=0)
        #
        # return np.median(mc_preds, axis=0), np.std(mc_preds, axis=0)
        probabilisticForecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(len(test))]
        for i in range(nc_simulation):
            for h in range(len(test)):
                lts_all[h].add_scenario(Scenario(np.mean(mc_preds, axis=0)[h], np.std(mc_preds, axis=0)[h]))
        for lts in lts_all:
            probabilisticForecast.add_variable(lts.scenarios, False)
        return probabilisticForecast
