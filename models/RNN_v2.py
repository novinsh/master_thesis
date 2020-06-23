import os
import sys

from models.model_base2 import ModelBase
from keras.models import Model, Input
from keras.layers import GRU, Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam
from utility.clr_callback import *
from utility.evaluation import *
from utility.probabilistic_forecast import *
from models.common import CustomGen, mean_loss, var_loss
from utility.visualization import plot_learning_curve
from tqdm.auto import tqdm


class ModelRNN(ModelBase):
    def __init__(self, name, n_input, n_features, batch_size=32, verbose=1):
        super(ModelRNN, self).__init__(name, n_input, n_features, batch_size, verbose)
        self.model = self.build_model(self.n_input, self.n_features, self.verbose)

    def build_model(self, n_input, n_features, verbose=1):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in, training=True)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        model = Model(inputs=x_in, outputs=mean, name=self.name)
        model.compile(loss='mean_squared_error', optimizer=Adam(0.01))
        print(model.summary()) if verbose else None
        return model

    def save_weights(self, path, overwrite=False):
        file_path = os.path.join(path, self.name)
        if os.path.isfile(file_path) and not overwrite:
            print("File already exists, choose another path or enforce overwrite!")
            assert False
        else:
            self.model.save_weights(file_path+ ".h5")
            print(f"Model weights saved in the desired path as: {self.name}.h5")

    def load_weights(self, file_path):
        if os.path.isfile(file_path):
            self.model.load_weights(file_path)
            print("Loaded the model weights from file!")
        else:
            print("File does not exists!")
            assert False


class ModelRNNNonBayesian(ModelRNN):
    def fit(self, train, dev, n_epochs=1, debug=False):
        assert self.n_features == train.shape[1]
        #
        generator = TimeseriesGenerator(train, train[:, 0], length=self.n_input)
        dev_generator = TimeseriesGenerator(dev, dev[:, 0], length=self.n_input)
        epoch_steps = train.shape[0] / self.batch_size
        #
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)
        self.history = self.model.fit_generator(generator,
                                                validation_data=dev_generator,
                                                shuffle=False, epochs=n_epochs, verbose=self.verbose,
                                                steps_per_epoch=epoch_steps, callbacks=[clr_cb])

    def forecast(self, warmup, test, nr_simulations=None):
        # TODO: assert length of warmup must be bigger or equal to n_input
        history = warmup[-self.n_input:].reshape(1, -1, 5)
        predictions = []
        ypred = self.model.predict(history)
        predictions.append(ypred[0][0])
        #
        for h in range(len(test) - 1):
            x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
            history = np.append(history, x_tp1, axis=1)
            history = history[:, -self.n_input:, :]
            ypred = self.model.predict(history)
            predictions.append(ypred[0][0])
        #
        predictions = np.array(predictions)
        return predictions

    def evaluate(self, warmup, test):
        ypred = self.forecast(warmup, test)
        ncrps, nmae = evaluate_nonquantile(test, ypred, period=len(test))
        mse = mean_squared_error(test, ypred)
        return mse, ncrps, nmae


class ModelRNNEpistemic(ModelRNNNonBayesian):
    def forecast(self, warmup, test, nr_simulations=10):
        horizon = len(test)

        def neuralnet_prediction(warmup, test):
            history = warmup.reshape(1, -1, 5)
            predictions = []
            ypred = self.model.predict(history)
            predictions.append(ypred[0][0])
            #
            for h in range(horizon - 1):
                x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
                history = np.append(history, x_tp1, axis=1)
                history = history[:, -self.n_input:, :]
                ypred = self.model.predict(history)
                predictions.append(ypred[0][0])
            #
            predictions = np.array(predictions)
            return predictions

        #
        mc_preds = np.empty((0, horizon))
        for i in range(nr_simulations):  # parallelize this part of the code
            ypred = neuralnet_prediction(warmup, test[:horizon])
            mc_preds = np.append(mc_preds, ypred.reshape(1, -1), axis=0)
        #
        # return np.median(mc_preds, axis=0), np.std(mc_preds, axis=0)
        probabilistic_forecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(horizon)]
        for i in range(nr_simulations):
            for h in range(horizon):
                lts_all[h].add_scenario(Scenario(np.mean(mc_preds, axis=0)[h], np.std(mc_preds, axis=0)[h]))
        for lts in lts_all:
            probabilistic_forecast.add_variable(lts.scenarios, False)
        return probabilistic_forecast

    def scenario_forecast(self, test_warmup, test, horizon=-1, n_sims=15):
        horizon = len(test) if horizon is None else horizon

        def neuralnet_scenario_forecasting(warmup, test, nr_simulations=1, debug=False):
            mcSimulation = MCSimulation(nr_simulations=nr_simulations)
            for i in tqdm(range(nr_simulations), desc="MCDO Simulation"):
                history = warmup.reshape(1, -1, warmup.shape[-1])
                trajectory = Trajectory(history=history, debug=debug)
                lts = LeadTimeScenarios()  # create new lead time scenarios
                for _ in range(5):
                    mus = []
                    for _ in range(10):
                        pred = self.model.predict(history)  ########################################
                        mu = pred[0][0]  # get the predicted mu and sigma
                        mus.append(mu)
                    mus = np.array(mus)
                    total_var = (mus ** 2).mean() - mus.mean() ** 2
                    lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                trajectory.add_leadtime(lts)  # add current scenario to the trajectory
                #
                for h in range(1, len(test)):  # roll forward
                    # generate histories based on previous LTS
                    histories = trajectory.generate_histories(n_input=self.n_input, n_samples=3, mode='01', test=test[h, 1:])
                    lts = LeadTimeScenarios()  # create new lead time scenario(s)
                    # traverse possible histories and predict one-step ahead which
                    # leads to traversing possible futures basically
                    for history in histories:
                        mus = []
                        for i in range(10):  # some number of times to get a variance from EP
                            pred = self.model.predict(history)  ################################
                            mu = pred[0][0]
                            mus.append(mu)
                        mus = np.array(mus)
                        total_var = (mus ** 2).mean() - mus.mean() ** 2
                        # print(history[0,:,0])
                        pred = self.model.predict(history)  ################################
                        lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                    # print("#"*10)
                    # Dropping some of the scenarios to avoid explosion of alternative futures
                    # lts.drop_randomly(probability=0.5)
                    trajectory.add_leadtime(lts)
                mcSimulation.add_trajectory(trajectory)
                # yhats = np.append(yhats, np.array(predictions).reshape(1,-1), axis=0)
                # yhats_std = np.append(yhats_std, np.sqrt(np.exp(predictions_logvar)).reshape(1,-1), axis=0)
            return mcSimulation

        # make prediction
        self.mcSimulation = neuralnet_scenario_forecasting(test_warmup, test[:horizon], n_sims, debug=True)

        def mcSimulation_to_probabilisticForecast(mcSimul, test):
            # print(len(mcSimulation.trajectories))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios[71].scenarios))
            probForecast = ProbabilisticForecast()
            assert n_sims == len(mcSimul.trajectories)
            #
            # aggregate results of MCDO and scenario forecasts:
            # we have a number of mu and sigmas per leadtime now (needs to be aggregated)
            lts_all = [LeadTimeScenarios() for _ in range(len(test))]  # aggregate all possible scenarios per time step
            for i in tqdm(range(n_sims),
                          desc='MCDO+ScenarioForecasting'):  # scenarios per lead time (aka leadtime scenarios) as a result of mcdo simulation
                for h, lts in enumerate(
                        mcSimul.trajectories[i].leadTimeScenarios):  # scenarios from scenario forecasting
                    for s in lts.scenarios:
                        lts_all[h].add_scenario(s)
            #
            for lts in tqdm(lts_all,
                            desc='LTS to RV'):  # create random variable for easier evaluation and visualization
                probForecast.add_variable(lts.scenarios, False)
            return probForecast

        probabilistic_forecast = mcSimulation_to_probabilisticForecast(self.mcSimulation, test[:horizon])
        return probabilistic_forecast


class ModelRNNAleatoric(ModelRNN):
    def __init__(self, name, n_input, n_features, batch_size=32, verbose=1):
        super(ModelRNNAleatoric, self).__init__(name, n_input, n_features, batch_size, verbose)

    def build_model(self, n_input, n_features, verbose=1):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        log_var = Dense(1, activation='linear', name='log_var')(x)
        model = Model(inputs=x_in, outputs=[mean, log_var], name=self.name)
        opt = Adam(lr=0.01) # TODO: set to same for all models
        model.compile(loss=[mean_loss(log_var), var_loss(log_var)], loss_weights=[.5, .5], optimizer=opt)
        print(model.summary()) if verbose else None
        return model

    def fit(self, train, dev, n_epochs=1, debug=False):
        assert self.n_features == train.shape[1]
        #
        generator = CustomGen(train, train[:, 0], length=self.n_input)
        dev_generator = CustomGen(dev, dev[:, 0], length=self.n_input)
        epoch_steps = train.shape[0] / self.batch_size
        #
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)
        self.history = self.model.fit_generator(generator,
                                                validation_data=dev_generator,
                                                shuffle=False, epochs=n_epochs, verbose=self.verbose,
                                                steps_per_epoch=epoch_steps, callbacks=[clr_cb])
        if debug:
            plot_learning_curve(self.history)

    def forecast(self, warmup, test, nr_simulations=10):
        history = warmup[-self.n_input:].reshape(1, -1, 5)
        # history = test_warmup[np.newaxis,-n_input:,:]# generator[-1][0][-1][np.newaxis,:,:]
        # print(history[:,-2:,:])
        predictions = []
        predictions_logvar = []
        ypred = self.model.predict(history)
        predictions.append(ypred[0][0][0])
        predictions_logvar.append(ypred[1][0][0])

        for h in range(len(test) - 1):
            x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
            history = np.append(history, x_tp1, axis=1)
            history = history[:, -self.n_input:, :]
            ypred = self.model.predict(history)
            predictions.append(ypred[0][0][0])
            predictions_logvar.append(ypred[1][0][0])

        predictions = np.array(predictions)
        predictions_std = np.sqrt(np.exp(predictions_logvar))
        # return predictions, predictions_std
        #
        probabilistic_forecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(len(test))]
        for h in range(len(test)):
            lts_all[h].add_scenario(Scenario(predictions[h], predictions_std[h]))
        for lts in lts_all:
            probabilistic_forecast.add_variable(lts.scenarios, False)
        return probabilistic_forecast


    def scenario_forecast(self, test_warmup, test, horizon=-1, n_sims=15):
        horizon = len(test) if horizon is None else horizon

        def neuralnet_scenario_forecasting(warmup, test, nr_simulations=1, debug=False):
            # yhats = np.empty((0,len(test)))
            # yhats_std = np.empty((0,len(test)))
            mcSimulation = MCSimulation(nr_simulations=nr_simulations)
            for i in tqdm(range(nr_simulations), desc="MCDO Simulation"):
                #         print(i)
                # Start a trajectory by warmup sequence and then generate new
                # alternative futures or aka scenarios based on new predictions
                history = warmup.reshape(1, -1, warmup.shape[-1])
                trajectory = Trajectory(history=history, debug=debug)
                # first lead time prediction, first scenario (only one)
                # TODO: could be more than only one, by running prediction multiple times (MCDO)
                lts = LeadTimeScenarios()  # create new lead time scenarios
                for _ in range(5):
                    mus = []
                    sigs = []
                    for _ in range(10):
                        pred = self.model.predict(history)  ########################################
                        mu, sigma = pred_to_musigma(pred)  # get the predicted mu and sigma
                        mus.append(mu)
                        sigs.append(sigma)
                    mus = np.array(mus)
                    sigs = np.array(sigs)
                    total_var = (sigs ** 2).mean()
                    lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                trajectory.add_leadtime(lts)  # add current scenario to the trajectory
                #
                for h in range(1, len(test)):  # roll forward
                    # generate histories based on previous LTS
                    histories = trajectory.generate_histories(n_input=self.n_input, n_samples=3, test=test[h, 1:])
                    lts = LeadTimeScenarios()  # create new lead time scenario(s)
                    # traverse possible histories and predict one-step ahead which
                    # leads to traversing possible futures basically
                    for history in histories:
                        mus = []
                        sigs = []
                        for _ in range(10):
                            pred = self.model.predict(history)  ########################################
                            mu, sigma = pred_to_musigma(pred)  # get the predicted mu and sigma
                            mus.append(mu)
                            sigs.append(sigma)
                    mus = np.array(mus)
                    sigs = np.array(sigs)
                    total_var = (sigs ** 2).mean()
                    lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                    # Dropping some of the scenarios to avoid explosion of alternative futures
                    # lts.drop_randomly(probability=0.5)
                    trajectory.add_leadtime(lts)
                mcSimulation.add_trajectory(trajectory)
                # yhats = np.append(yhats, np.array(predictions).reshape(1,-1), axis=0)
                # yhats_std = np.append(yhats_std, np.sqrt(np.exp(predictions_logvar)).reshape(1,-1), axis=0)
            return mcSimulation

        # make prediction
        self.mcSimulation = neuralnet_scenario_forecasting(test_warmup, test[:horizon], n_sims, debug=True)

        def mcSimulation_to_probabilisticForecast(mcSimul, test):
            # print(len(mcSimulation.trajectories))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios[71].scenarios))
            probForecast = ProbabilisticForecast()
            assert n_sims == len(mcSimul.trajectories)
            #
            # aggregate results of MCDO and scenario forecasts:
            # we have a number of mu and sigmas per leadtime now (needs to be aggregated)
            lts_all = [LeadTimeScenarios() for _ in range(len(test))]  # aggregate all possible scenarios per time step
            for i in tqdm(range(n_sims),
                          desc='MCDO+ScenarioForecasting'):  # scenarios per lead time (aka leadtime scenarios) as a result of mcdo simulation
                for h, lts in enumerate(
                        mcSimul.trajectories[i].leadTimeScenarios):  # scenarios from scenario forecasting
                    for s in lts.scenarios:
                        lts_all[h].add_scenario(s)
            #
            for lts in tqdm(lts_all,
                            desc='LTS to RV'):  # create random variable for easier evaluation and visualization
                probForecast.add_variable(lts.scenarios, False)
            return probForecast

        probabilistic_forecast = mcSimulation_to_probabilisticForecast(self.mcSimulation, test[:horizon])
        return probabilistic_forecast


class ModelRNNAleatoricEpistemic(ModelRNNAleatoric):
    def forecast(self, warmup, test, nr_simulations=10):
        yhats = np.empty((0, len(test)))
        yhats_std = np.empty((0, len(test)))
        for i in range(nr_simulations):
            history = warmup.reshape(1, -1, warmup.shape[-1])
            predictions = []
            predictions_logvar = []
            ypred = self.model.predict(history)
            predictions.append(ypred[0][0][0])
            predictions_logvar.append(ypred[1][0][0])
            for h in range(len(test) - 1):
                x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
                history = np.append(history, x_tp1, axis=1)
                history = history[:, -self.n_input:, :]
                ypred = self.model.predict(history)
                predictions.append(ypred[0][0][0])
                predictions_logvar.append(ypred[1][0][0])
            yhats = np.append(yhats, np.array(predictions).reshape(1, -1), axis=0)
            yhats_std = np.append(yhats_std, np.sqrt(np.exp(predictions_logvar)).reshape(1, -1), axis=0)
        #
        probabilistic_forecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(len(test))]
        for i in range(nr_simulations):
            for h in range(len(test)):
                lts_all[h].add_scenario(Scenario(yhats[i, h], yhats_std[i, h]))
        for lts in lts_all:
            probabilistic_forecast.add_variable(lts.scenarios, False)
        return probabilistic_forecast

    def scenario_forecast(self, test_warmup, test, horizon=-1, n_sims=15):
        horizon = len(test) if horizon is None else horizon

        def neuralnet_scenario_forecasting(warmup, test, nr_simulations=1, debug=False):
            # yhats = np.empty((0,len(test)))
            # yhats_std = np.empty((0,len(test)))
            mcSimulation = MCSimulation(nr_simulations=nr_simulations)
            for i in tqdm(range(nr_simulations), desc="MCDO Simulation"):
                #         print(i)
                # Start a trajectory by warmup sequence and then generate new
                # alternative futures or aka scenarios based on new predictions
                history = warmup.reshape(1, -1, warmup.shape[-1])
                trajectory = Trajectory(history=history, debug=debug)
                # first lead time prediction, first scenario (only one)
                # TODO: could be more than only one, by running prediction multiple times (MCDO)
                lts = LeadTimeScenarios()  # create new lead time scenarios
                for _ in range(5):
                    mus = []
                    sigs = []
                    for _ in range(10):
                        pred = self.model.predict(history)  ########################################
                        mu, sigma = pred_to_musigma(pred)  # get the predicted mu and sigma
                        mus.append(mu)
                        sigs.append(sigma)
                    mus = np.array(mus)
                    sigs = np.array(sigs)
                    total_var = (mus ** 2).mean() - mus.mean() ** 2 + (sigs ** 2).mean()
                    lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                trajectory.add_leadtime(lts)  # add current scenario to the trajectory
                #
                for h in range(1, len(test)):  # roll forward
                    # generate histories based on previous LTS
                    histories = trajectory.generate_histories(n_input=self.n_input, n_samples=3, test=test[h, 1:])
                    lts = LeadTimeScenarios()  # create new lead time scenario(s)
                    # traverse possible histories and predict one-step ahead which
                    # leads to traversing possible futures basically
                    for history in histories:
                        mus = []
                        sigs = []
                        for i in range(10): # some number of times to get a variance from EP
                            pred = self.model.predict(history)  ################################
                            mu, sigma = pred_to_musigma(pred)
                            mus.append(mu)
                            sigs.append(sigma)
                        mus = np.array(mus)
                        sigs = np.array(sigs)
                        total_var = (mus**2).mean() - mus.mean()**2 + (sigs**2).mean()
                        lts.add_scenario(Scenario(np.mean(mus), np.sqrt(total_var)))  # create new scenario
                    # Dropping some of the scenarios to avoid explosion of alternative futures
                    # lts.drop_randomly(probability=0.5)
                    trajectory.add_leadtime(lts)
                mcSimulation.add_trajectory(trajectory)
                # yhats = np.append(yhats, np.array(predictions).reshape(1,-1), axis=0)
                # yhats_std = np.append(yhats_std, np.sqrt(np.exp(predictions_logvar)).reshape(1,-1), axis=0)
            return mcSimulation

        # make prediction
        self.mcSimulation = neuralnet_scenario_forecasting(test_warmup, test[:horizon], n_sims, debug=True)

        def mcSimulation_to_probabilisticForecast(mcSimul, test):
            # print(len(mcSimulation.trajectories))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios))
            # print(len(mcSimulation.trajectories[9].leadTimeScenarios[71].scenarios))
            probForecast = ProbabilisticForecast()
            assert n_sims == len(mcSimul.trajectories)
            #
            # aggregate results of MCDO and scenario forecasts:
            # we have a number of mu and sigmas per leadtime now (needs to be aggregated)
            lts_all = [LeadTimeScenarios() for _ in range(len(test))]  # aggregate all possible scenarios per time step
            for i in tqdm(range(n_sims),
                          desc='MCDO+ScenarioForecasting'):  # scenarios per lead time (aka leadtime scenarios) as a result of mcdo simulation
                for h, lts in enumerate(
                        mcSimul.trajectories[i].leadTimeScenarios):  # scenarios from scenario forecasting
                    for s in lts.scenarios:
                        lts_all[h].add_scenario(s)
            #
            for lts in tqdm(lts_all,
                            desc='LTS to RV'):  # create random variable for easier evaluation and visualization
                probForecast.add_variable(lts.scenarios, False)
            return probForecast

        probabilistic_forecast = mcSimulation_to_probabilisticForecast(self.mcSimulation, test[:horizon])
        return probabilistic_forecast
