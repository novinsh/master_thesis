import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_evaluation(ncrps, nmae, period=24, title=''):
    plt.figure(figsize=(15, 5))
    plt.errorbar(range(1, period + 1), np.mean(ncrps, axis=(0, 2)), 
                 yerr=np.std(np.mean(ncrps, axis=0), axis=1), color='red', fmt='o-', capsize=3,
                 label='NCRPS', markersize=1, elinewidth=1)
    plt.errorbar(range(1, period + 1), np.mean(nmae, axis=(0, 2)), 
                 yerr=np.std(np.mean(nmae, axis=0), axis=1), color='blue', fmt='o-', capsize=3,
                 label='NMAE', markersize=1, elinewidth=1)
    plt.hlines(0, 1, period + 1, color='k', linestyles=':', linewidth=1.5)
    plt.legend(fontsize=15)
    plt.xlabel('Lead time', fontsize=15)
    plt.grid()
    plt.xticks(range(1, period + 1), range(1, period + 1))
    plt.title(title)
    #plt.ylim([0,0.2])
    plt.show()


def plot_evaluation_2(ncrps, nmae, period=24, title=''):
    plt.figure(figsize=(15, 5))
    plt.plot(range(1, period + 1), np.mean(ncrps, axis=(0, 2)),color='red',label='NCRPS')
    plt.plot(range(1, period + 1), np.mean(nmae, axis=(0, 2)),color='blue',label='NMAE')
    plt.legend(fontsize=15)
    plt.xlabel('Lead time', fontsize=15)
    plt.grid()
    plt.xticks(range(1, period + 1), range(1, period + 1))
    plt.title(title)
    # plt.ylim([-6, 2])
    plt.show()


def plot_evaluation_3(ncrps, nmse, bounds, period=24, title=''):
    """ with RMSE and bounds """
    plt.figure(figsize=(15, 5))
    plt.errorbar(range(1, period + 1), np.mean(ncrps, axis=(0, 2)),
                 yerr=np.std(np.mean(ncrps, axis=0), axis=1), color='red', fmt='o-', capsize=3,
                 label='NCRPS', markersize=1, elinewidth=1)
    plt.errorbar(range(1, period + 1), np.mean(nmse, axis=(0, 2)),
                 yerr=np.std(np.mean(nmse, axis=0), axis=1), color='blue', fmt='o-', capsize=3,
                 label='NMSE', markersize=1, elinewidth=1)
    ubound, lbound1, lbound2, lbound3 = bounds
    plt.hlines(ubound, 0, period  + 1, color='m', linestyles='--', label='upper bound')
    plt.hlines(lbound3, 0, period  + 1, color='g', linestyles='--', label='lower bound')
    plt.hlines(0, 1, period + 1, color='k', linestyles=':', linewidth=1.5)
    plt.legend(fontsize=15)
    plt.xlabel('Lead time', fontsize=15)
    plt.grid()
    plt.xticks(range(1, period + 1), range(1, period + 1))
    plt.title(title)
    #plt.ylim([0,0.2])
    plt.show()


def plot_forecast_dense(x, y, forecast, horizon=24, title="", baselines=None, alphas=[0.1, 0.2, 0.3, 0.4, 0.5], figsize=(17,4)):
    plt.figure(figsize=figsize)
    offset_te = min(horizon, 72) # how much of the test data to plot, not more than 48!
    offset_tr = 24#int(0.5 * offset_te)  # how much of training data to plot
    #
    plt.plot(range(len(x) - offset_tr, len(x) + horizon), np.append(x[-offset_tr:], y), ".-k", zorder=3, label='Ground truth')
    plt.vlines(len(x), 0, 1, linestyles="--", color="grey", label='train-test')
    #
    k=0
    opacities = np.arange(.5/len(alphas),0.9,0.05)
    for i, alpha in enumerate(alphas):
        yhat_interval = forecast.interval(alpha)
        plt.fill_between(range(len(x), len(x) + horizon), yhat_interval[0], yhat_interval[1], 
                         alpha=opacities[i], color="b", label="{0:2.0f}% PI".format((1-alpha)*100), linewidth=0.0)
    plt.plot(range(len(x), len(x) + horizon), 
             forecast.median(), "--b", zorder=2, linewidth=2, label="median")
    plt.plot(range(len(x), len(x) + horizon), 
             forecast.mean(), "-b", zorder=2, linewidth=2, label="mean")
    #
    if baselines:
        colors = ['m', 'c', 'g', 'y'] # TODO: better coloring
        for i, (name, prediction) in enumerate(baselines.items()):
            plt.plot(range(len(x), len(x) + horizon), prediction, colors[i], 
                     zorder=1, linewidth=2, linestyle='-.', label=name)
    #
    print(offset_te)
    plt.xlim([len(x) - offset_tr, len(x) + offset_te - 1])
    plt.ylim([0, 1])
    plt.xlabel('Lead time')
    plt.ylabel('Power')
    xticks = np.arange(len(x) - offset_tr, len(x) + offset_te, 4)
    xtick_labels = np.arange(-offset_tr, offset_te, 4).astype('str')
    if horizon > offset_te:
        xticks = np.append(xticks, [len(x)+offset_te-1])
        xtick_labels = np.append(xtick_labels,['...'])
    plt.grid(alpha=0.25, which='major', linestyle='--')
    plt.xticks(xticks, xtick_labels)
    yticks = np.round(np.linspace(0,1,10+1),1)
    plt.yticks(yticks, yticks)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.show()


def plot_forecast_dense_2(x, y, forecast, horizon=24, title="", baselines=None, alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
    plt.figure(figsize=(17, 4))
    offset_te = min(horizon, 72) # how much of the test data to plot, not more than 48!
    offset_tr = 24#int(0.5 * offset_te)  # how much of training data to plot
    #
    plt.plot(range(len(x) - offset_tr, len(x) + horizon), np.append(x[-offset_tr:], y), ".-k", zorder=3, label='Ground truth')
    plt.vlines(len(x), 0, 1, linestyles="--", color="grey", label='train-test')
    #
    k=0
    opacities = np.arange(.3/len(alphas),0.7,0.025)
    for i, alpha in enumerate(alphas):
        yhat_interval = forecast.interval(alpha)
        fac = (i/2+1) if i > 10 else (i/5+1)
        if i == len(alphas)-1:
            plt.fill_between(range(len(x), len(x) + horizon), yhat_interval[0], yhat_interval[1],
                             alpha=opacities[i] / fac, color="b", linewidth=0.0, label="Dense Forecast")
        else:
            plt.fill_between(range(len(x), len(x) + horizon), yhat_interval[0], yhat_interval[1],
                             alpha=opacities[i]/fac, color="b", linewidth=0.0)
    plt.plot(range(len(x), len(x) + horizon),
             forecast.median(), "--b", zorder=2, linewidth=2, label="median")
    plt.plot(range(len(x), len(x) + horizon),
             forecast.mean(), "-b", zorder=2, linewidth=2, label="mean")
    #
    if baselines:
        colors = ['r', 'm', 'c', 'g', 'y'] # TODO: better coloring
        for i, (name, prediction) in enumerate(baselines.items()):
            plt.plot(range(len(x), len(x) + horizon), prediction, colors[i],
                     zorder=1, linewidth=2, linestyle='-.', label=name)
    #
    print(offset_te)
    plt.xlim([len(x) - offset_tr, len(x) + offset_te - 1])
    plt.ylim([0, 1])
    plt.xlabel('Lead time')
    plt.ylabel('Power')
    xticks = np.arange(len(x) - offset_tr, len(x) + offset_te, 4)
    xtick_labels = np.arange(-offset_tr, offset_te, 4).astype('str')
    if horizon > offset_te:
        xticks = np.append(xticks, [len(x)+offset_te-1])
        xtick_labels = np.append(xtick_labels,['...'])
    plt.grid(alpha=0.25, which='major', linestyle='--')
    plt.xticks(xticks, xtick_labels)
    yticks = np.round(np.linspace(0,1,10+1),1)
    plt.yticks(yticks, yticks)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.show()

def plot_forecast_quantile(x, y, forecast, horizon=24, title="", baselines=None, alphas=[0.1, 0.2, 0.3, 0.4, 0.5]):
    plt.figure(figsize=(17, 4))
    offset_te = min(horizon, 72) # how much of the test data to plot, not more than 48!
    offset_tr = 24#int(0.5 * offset_te)  # how much of training data to plot
    #
    plt.plot(range(len(x) - offset_tr, len(x) + horizon), np.append(x[-offset_tr:], y), ".-k", zorder=3, label='Ground truth')
    plt.vlines(len(x), 0, 1, linestyles="--", color="grey", label='train-test')
    #
    k=0
    opacities = np.arange(.5/len(alphas),0.9,0.05)
    for i, alpha in enumerate(alphas):
        yhat_interval = forecast.interval(alpha)
        # plt.fill_between(range(len(x), len(x) + horizon), yhat_interval[0], yhat_interval[1],
        #                  alpha=opacities[i], color="b", linewidth=0.0)
        plt.plot(range(len(x), len(x) + horizon),
                 yhat_interval[0], "-b1", zorder=2, linewidth=1.75)
        plt.plot(range(len(x), len(x) + horizon),
                 yhat_interval[1], "-b1", zorder=2, linewidth=1.75, label="{0:2.0f}% Quantile".format((1-alpha)*100))

    plt.plot(range(len(x), len(x) + horizon),
             forecast.median(), "--b", zorder=2, linewidth=2, label="median")
    plt.plot(range(len(x), len(x) + horizon),
             forecast.mean(), "-b", zorder=2, linewidth=2, label="mean")
    #
    if baselines:
        colors = ['-r', '-m', 'c', 'g', 'y'] # TODO: better coloring
        for i, (name, prediction) in enumerate(baselines.items()):
            plt.plot(range(len(x), len(x) + horizon), prediction, colors[i],
                     zorder=1, linewidth=2, linestyle=':', label=name)
    #
    print(offset_te)
    plt.xlim([len(x) - offset_tr, len(x) + offset_te - 1])
    plt.ylim([0, 1])
    plt.xlabel('Lead time')
    plt.ylabel('Power')
    xticks = np.arange(len(x) - offset_tr, len(x) + offset_te, 4)
    xtick_labels = np.arange(-offset_tr, offset_te, 4).astype('str')
    if horizon > offset_te:
        xticks = np.append(xticks, [len(x)+offset_te-1])
        xtick_labels = np.append(xtick_labels,['...'])
    plt.grid(alpha=0.25, which='major', linestyle='--')
    plt.xticks(xticks, xtick_labels)
    yticks = np.round(np.linspace(0,1,10+1),1)
    plt.yticks(yticks, yticks)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.show()


def plot_mcSims(mcSimulation, horizon):
    # only works during debug runtime
    plt.figure(figsize=(17,3))
    ax = plt.axes()
    for h in range(horizon):
        for sc in mcSimulation.trajectories[0].leadTimeScenarios[h].scenarios:
            ax.plot(h, sc.mu, 'r.')
            if h != horizon-1:
                #print(mcSimulation.trajectories[0].histories_track[h])
                samples = mcSimulation.trajectories[0].histories_track[h]
                nr_samples = len(samples)
                ax.plot([h+0.5]*nr_samples, samples, 'b')
                #for history in mcSimulation.trajectories[0].histories_track[h]:
                #    ax.arrow(h, sc.mu, h+0.5, history, color='b')
    ax.plot([],[], 'r.', label='predicted gaussian')            
    ax.plot([],[], 'b*', label='sampled from the prediction gaussian')
    plt.title('gaussian predictions and samples for creating new trajectories')
    plt.legend()
    plt.show()


def plot_mcBranches():
	pass
	# TODO: visualize the trajectories in a tree manner
	# ax = plt.axes()
	# for h in range(horizon):
	#     if h < 1:
	#         continue
	#     vt = mcSimulation.trajectories[0].evolved_trajectory[h]
	#     vtm1 = mcSimulation.trajectories[0].evolved_trajectory[h-1]
	#     print(vt)
	#     print(vtm1)
	#     for i in range(2):
	#         pass
	#         #for j in range(len(vt)):
	#         #ax.arrow(h-1, vtm1[i], h, vt[j,i])
	#     break
	#     for v in evj: 
	# #         print(v)
	#         if h > 0:
	#             ax.arrow(h, v[0], h, v[1], color='blue', alpha=0.25)
	#     if h > 10:
	#         break
	# plt.show()


def plot_learning_curve(history):
    plt.figure(figsize=(12,3))
    plt.subplot(1,3,1)
    # Learning curve
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['train', 'validation'])
    plt.xlabel('epoch')
    plt.title('total loss')
    plt.grid()
    # plt.yscale('log')

    if 'mean_loss' in history.history.keys() and 'val_mean_loss' in history.history.keys():
        plt.subplot(1,3,2)
        plt.plot(history.history['mean_loss'])
        plt.plot(history.history['val_mean_loss'])
        plt.legend(['train', 'validation'])
        plt.xlabel('epoch')
        plt.title('loss 1')
        plt.grid()
        # plt.yscale('log')

    if 'log_var_loss' in history.history.keys() and 'val_log_var_loss' in history.history.keys():
        plt.subplot(1,3,3)
        plt.plot(history.history['log_var_loss'])
        plt.plot(history.history['val_log_var_loss'])
        plt.legend(['train', 'validation'])
        # plt.yscale('log')
        plt.xlabel('epoch')
        plt.title('loss 2')
        plt.grid()
    plt.show()


def plot_vis1(yhats, yhats_std, test, expressive=False):
    """ visualization 1 """
    aleatoric_std = np.median(yhats_std, axis=0)
    yhats_median = np.median(yhats, axis=0)
    total_error = np.sqrt(np.var(yhats, axis=0) + aleatoric_std ** 2)

    def plot_diff():
        # see the difference between aleatoric and epistemic
        plt.plot(total_error, label='epistemic+aleatoric')
        plt.plot(aleatoric_std, label='aleatoric')
        plt.legend()
        plt.show()

    plt.plot(test[:, 0], '.-', label='ground truth')
    plt.plot(yhats_median, '.-', label='prediction median')
    plt.errorbar(range(yhats.shape[1]), yhats_median, yerr=np.std(yhats, axis=0), color='red', fmt='o', capsize=3,
                 label='MCDO Error (epistemic)', markersize=1, elinewidth=1)
    plt.fill_between(range(len(test)), yhats_median - aleatoric_std,
                     yhats_median + aleatoric_std, color='orange', alpha=0.25, label='Predictive Error (aleatoric)')
    plt.fill_between(range(len(test)), yhats_median - total_error,
                     yhats_median + total_error, color='green', alpha=0.25, label='PI (aleatoric+epistemic)')
    # plt.plot(dev_sig[-n_input:], label='without noise')
    plt.title("MSE: %2.3f  MCDO: %2.3f Aleatoric: %2.3f" %
              (mean_squared_error(test, yhats_median),
               np.mean(np.std(yhats, axis=0)),
               np.mean(aleatoric_std)
               )
              )
    plt.ylim([0, 1])
    plt.legend(loc='best')
    plt.show()
    plot_diff() if expressive else None


def plot_vis2(yhats, yhats_std, test, expressive=False):
    """ visualization 2 """

    def plott_std(std_min, std_med, std_max):
        # see the difference between min, max and median of the MCDO standard deviation
        plt.plot(std_max, label='$\sigma_{max}$')
        plt.plot(std_med, label='$\sigma_{median}$ ')
        plt.plot(std_min, label='$\sigma_{min}$ ')
        plt.legend()
        plt.show()

    std_min = np.min(yhats_std, axis=0)
    std_med = np.median(yhats_std, axis=0)
    std_max = np.max(yhats_std, axis=0)
    yhats_median = np.median(yhats, axis=0)
    #
    idx = np.random.choice(np.arange(len(yhats)), 20)
    plt.plot(yhats[idx, :].T, 'b-', alpha=0.25)
    plt.plot([], [], label='MCDO predictions')
    plt.fill_between(range(yhats.shape[1]), yhats_median - std_max,
                     yhats_median + std_max, color='blue', alpha=0.05, label='std max')
    plt.fill_between(range(yhats.shape[1]), yhats_median - std_med,
                     yhats_median + std_med, color='blue', alpha=0.15, label='std median')
    plt.fill_between(range(yhats.shape[1]), yhats_median - std_min,
                     yhats_median + std_min, color='blue', alpha=0.25, label='std min')
    plt.legend()
    plt.show()
    plott_std(std_min, std_med, std_max) if expressive else None