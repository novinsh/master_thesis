from models.model_base import ModelBase
from keras.models import Model, Input
from keras.layers import GRU, Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from utility.clr_callback import *
from utility.evaluation import *


class ModelNonBayesian(ModelBase):
    def build_model(self, n_input, n_features):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        model = Model(inputs=x_in, outputs=mean)
        model.compile(loss='mean_squared_error', optimizer='adam')
        print(model.summary())
        return model

    def fit(self, train, dev, n_input=24, debug=True):
        n_features = train.shape[1]
        #
        generator = TimeseriesGenerator(train, train[:, 0], length=n_input)
        dev_generator = TimeseriesGenerator(dev, dev[:, 0], length=n_input)
        batch_size = 32
        epoch_steps = train.shape[0] / batch_size
        #
        self.model0 = self.build_model(n_input, n_features)
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)
        history = self.model0.fit_generator(generator, validation_data=dev_generator, shuffle=False,
                                            epochs=10, verbose=1, steps_per_epoch=epoch_steps, callbacks=[clr_cb])
        self.history = history
        self.n_input = n_input

    def forecast(self, warmup, test):
        n_input = self.n_input
        history = warmup[-n_input:].reshape(1, -1, 5)
        predictions = []
        ypred = self.model0.predict(history)
        predictions.append(ypred[0][0])
        #
        for h in range(len(test) - 1):
            x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
            history = np.append(history, x_tp1, axis=1)
            history = history[:, -n_input:, :]
            ypred = self.model0.predict(history)
            predictions.append(ypred[0][0])
        #
        predictions = np.array(predictions)
        return predictions

    def evaluate(self, warmup, test):
        ypred = self.forecast(warmup, test)
        ncrps, nmae = evaluate_nonquantile(test, ypred, period=lest(test))
        mse = mean_squared_error(test, ypred)
        return mse, ncrps, nmae

