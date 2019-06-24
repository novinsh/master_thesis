import keras
from models.model_base import ModelBase
from models.RNN.common import *
from keras.models import Model, Input
from keras.layers import GRU, Dense
from utility.clr_callback import *
from utility.visualization import *
from utility.probabilistic_forecast import *
from tqdm.autonotebook import tqdm


class ModelAleatoricEpistemic(ModelBase):
    def build_model(self, n_input, n_features):
        #
        x_in = Input(shape=(n_input, n_features,))
        x = GRU(100, recurrent_dropout=0.3, return_sequences=False, input_shape=(n_input, n_features))(x_in,
                                                                                                       training=True)
        mean = Dense(1, activation='hard_sigmoid', name='mean')(x)
        log_var = Dense(1, activation='linear', name='log_var')(x)
        model = Model(inputs=x_in, outputs=[mean, log_var])
        opt = keras.optimizers.Adam(lr=0.01)
        model.compile(loss=[mean_loss(log_var), var_loss(log_var)], loss_weights=[.5, .5], optimizer=opt)
        print(model.summary())
        return model

    def fit(self, train, dev, n_input=24, debug=False):
        n_features = train.shape[1]

        generator = CustomGen(train, train[:, 0], length=n_input)
        dev_generator = CustomGen(dev, dev[:, 0], length=n_input)
        batch_size = 32
        epoch_steps = train.shape[0] / batch_size

        self.model_alep = self.build_model(n_input, n_features)
        clr_cb = CyclicLR(mode='triangular2', base_lr=0.00001, max_lr=0.01, step_size=epoch_steps, gamma=0.8)
        self.history = self.model_alep.fit_generator(generator, validation_data=dev_generator, shuffle=False,
                                                     epochs=10, verbose=1, steps_per_epoch=epoch_steps,
                                                     callbacks=[clr_cb])
        if debug:
            plot_learning_curve(self.history)
        self.n_input = n_input

    def forecast(self, warmup, test, nr_simulations=10):
        n_input = self.n_input
        yhats = np.empty((0, len(test)))
        yhats_std = np.empty((0, len(test)))
        for i in range(nr_simulations):
            history = warmup.reshape(1, -1, 5)
            predictions = []
            predictions_logvar = []
            ypred = self.model_alep.predict(history)
            predictions.append(ypred[0][0][0])
            predictions_logvar.append(ypred[1][0][0])
            for h in range(len(test) - 1):
                x_tp1 = np.append(predictions[-1], test[h, 1:]).reshape(1, 1, -1)
                history = np.append(history, x_tp1, axis=1)
                history = history[:, -n_input:, :]
                ypred = self.model_alep.predict(history)
                predictions.append(ypred[0][0][0])
                predictions_logvar.append(ypred[1][0][0])
            yhats = np.append(yhats, np.array(predictions).reshape(1, -1), axis=0)
            yhats_std = np.append(yhats_std, np.sqrt(np.exp(predictions_logvar)).reshape(1, -1), axis=0)
        #
        probabilisticForecast = ProbabilisticForecast()
        lts_all = [LeadTimeScenarios() for _ in range(len(test))]
        for i in range(nr_simulations):
            for h in range(len(test)):
                lts_all[h].add_scenario(Scenario(yhats[i, h], yhats_std[i, h]))
        for lts in lts_all:
            probabilisticForecast.add_variable(lts.scenarios, False)
        return probabilisticForecast

    def scenario_forecast(self, test_warmup, test, horizon=24, n_sims=15):
        n_input = self.n_input

        def neuralnet_scenario_forecasting(warmup, test, nr_simulations=1, debug=False):
            # yhats = np.empty((0,len(test)))
            # yhats_std = np.empty((0,len(test)))
            mcSimulation = MCSimulation(nr_simulations=nr_simulations)
            for i in tqdm(range(nr_simulations), desc="MCDO Simulation"):
                #         print(i)
                # Start a trajectory by warmup sequence and then generate new
                # alternative futures or aka scenarios based on new predictions
                history = warmup.reshape(1, -1, 5)
                trajectory = Trajectory(history=history, debug=debug)
                # first lead time prediction, first scenario (only one)
                # TODO: could be more than only one, by running prediction multiple times (MCDO)
                lts = LeadTimeScenarios()  # create new lead time scenarios
                for _ in range(10):
                    pred = self.model_alep.predict(history)  ########################################
                    mu, sigma = pred_to_musigma(pred)  # get the predicted mu and sigma
                    lts.add_scenario(Scenario(mu, sigma))  # create new scenario
                trajectory.add_leadtime(lts)  # add current scenario to the trajectory
                #
                for h in range(1, len(test)):  # roll forward
                    # generate histories based on previous LTS
                    histories = trajectory.generate_histories(n_input=n_input, n_samples=3, test=test[h, 1:])
                    lts = LeadTimeScenarios()  # create new lead time scenario(s)
                    # traverse possible histories and predict one-step ahead which
                    # leads to traversing possible futures basically
                    for history in histories:
                        pred = self.model_alep.predict(history)  ################################
                        mu, sigma = pred_to_musigma(pred)
                        lts.add_scenario(Scenario(mu, sigma))  # create new scenario
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

        probabilisticForecast = mcSimulation_to_probabilisticForecast(self.mcSimulation, test[:horizon])
        return probabilisticForecast