import keras
from models.model_base import ModelBase
from models.common import *
from keras.models import Model, Input
from keras.layers import GRU, Dense
from utility.clr_callback import *
from utility.visualization import *
from utility.probabilistic_forecast import *


class ModelAleatoric(ModelBase):
    def build_model(self, n_input, n_features):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        log_var = Dense(1, activation='linear', name='log_var')(x)
        model = Model(inputs=x_in, outputs=[mean, log_var])
        opt = keras.optimizers.Adam(lr=0.01)
        model.compile(loss=[mean_loss(log_var), var_loss(log_var)], loss_weights=[.5, .5], optimizer=opt)
        print(model.summary())
        return model

    def fit(self, train, dev, n_input=24, nr_epochs=10, debug=False):
        n_features = train.shape[1]

        generator = CustomGen(train, train[:, 0], length=n_input)
        dev_generator = CustomGen(dev, dev[:, 0], length=n_input)
        batch_size = 32
        epoch_steps = train.shape[0] / batch_size

        self.model_al = self.build_model(n_input, n_features)
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)

        self.history = self.model_al.fit_generator(generator, validation_data=dev_generator, shuffle=False,
                                                   epochs=nr_epochs, verbose=1, steps_per_epoch=epoch_steps,
                                                   callbacks=[clr_cb])
        self.n_input = n_input
        if debug:
            plot_learning_curve(self.history)

    def forecast(self, warmup, test):
        n_input = self.n_input
        history = warmup[-n_input:].reshape(1, -1, 5)
        # history = test_warmup[np.newaxis,-n_input:,:]# generator[-1][0][-1][np.newaxis,:,:]
        # print(history[:,-2:,:])
        predictions = []
        predictions_logvar = []
        ypred = self.model_al.predict(history)
        predictions.append(ypred[0][0][0])
        predictions_logvar.append(ypred[1][0][0])

        for h in range(len(test) - 1):
            x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
            history = np.append(history, x_tp1, axis=1)
            history = history[:, -n_input:, :]
            ypred = self.model_al.predict(history)
            predictions.append(ypred[0][0][0])
            predictions_logvar.append(ypred[1][0][0])

        predictions = np.array(predictions)
        predictions_std = np.sqrt(np.exp(predictions_logvar))
        # return predictions, predictions_std
        #
        probabilisticForecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(len(test))]
        for h in range(len(test)):
            lts_all[h].add_scenario(Scenario(predictions[h], predictions_std[h]))
        for lts in lts_all:
            probabilisticForecast.add_variable(lts.scenarios, False)
        return probabilisticForecast