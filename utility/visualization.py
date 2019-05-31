import numpy as np
from matplotlib import pyplot as plt
from models.model import Model


def plot_forecast_dense(x, y, y_pred, horizon=24, title="", baselines=None, quantiles=[0.05, 0.10, 0.15, 0.20, 0.25]):
    plt.figure(figsize=(17, 4))
    offset_te = min(horizon, 48) # how much of the test data to plot, not more than 48!
    offset_tr = int(0.5 * offset_te)  # how much of training data to plot
    #
    plt.plot(range(len(x) - offset_tr, len(x) + horizon), np.append(x[-offset_tr:], y), ".-k", zorder=3)
    plt.vlines(len(x), 0, 1, linestyles="--", color="grey", label='train-test')
    #
    median = Model.get_horizon(y_pred, .5)
    mean = (Model.get_horizon(y_pred, 0.01) + Model.get_horizon(y_pred, 0.99)) / 2
    k=0
    alphas = np.arange(.5/len(quantiles),0.9,0.05)
    for i, q in enumerate(quantiles):
        yhat_q = Model.get_horizon(y_pred, q)[:,k], Model.get_horizon(y_pred, 1-q)[:,k]
        plt.fill_between(range(len(x), len(x) + horizon), yhat_q[0], yhat_q[1], alpha=alphas[i], color="b", label="{0}% PI".format((1-2*q)*100), linewidth=0.0)
    plt.plot(range(len(x), len(x) + horizon), median, "--b", zorder=2, label="median")
    plt.plot(range(len(x), len(x) + horizon), mean, "-b", zorder=2, label="mean")
    #
    if baselines:
        colors = ['-r', '-m', 'c', 'g', 'y'] # TODO: better coloring
        for i, (name, prediction) in enumerate(baselines.items()):
            plt.plot(range(len(x), len(x) + horizon), prediction, colors[i], zorder=1, linestyle=':', label=name)
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
    plt.xticks(xticks, xtick_labels)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    plt.show()


# def plot_forecast_dense(x, y, y_pred, horizon=24, title=""):
#     plt.figure(figsize=(17, 4))
#     offset = 1 * horizon  # how much of training data to plot
#     #     plt.plot(range(len(x)-offset,len(x)), x[-offset:], ".-k")
#     #     plt.plot(range(len(x),len(x)+horizon), y, ".-k")
#     print(horizon)
#     plt.plot(range(len(x) - offset, len(x) + horizon), np.append(x[-offset:], y), ".-k", zorder=3)
#     plt.vlines(len(x), 0, 1, linestyles="--", color="grey", label='train-test')
#     plt.plot(range(len(x), len(x) + horizon), y_pred[:, 50, 0], "--b", zorder=2, label="median")
#     plt.plot(range(len(x), len(x) + horizon), np.mean(y_pred[:, :, 0], axis=1), "-b", zorder=2, label="mean")
#     # plt.fill_between(range(len(x), len(x) + horizon), y_pred[:, 5, 0], y_pred[:, 95, 0], alpha=0.1, color="b",
#     #                  label="90% PI", linewidth=0.0)
#     # plt.fill_between(range(len(x), len(x) + horizon), y_pred[:, 10, 0], y_pred[:, 90, 0], alpha=0.15, color="b",
#     #                  label="80% PI", linewidth=0.0)
#     # plt.fill_between(range(len(x), len(x) + horizon), y_pred[:, 15, 0], y_pred[:, 85, 0], alpha=0.2, color="b",
#     #                  label="70% PI", linewidth=0.0)
#     # plt.fill_between(range(len(x), len(x) + horizon), y_pred[:, 20, 0], y_pred[:, 80, 0], alpha=0.25, color="b",
#     #                  label="60% PI", linewidth=0.0)
#     plt.fill_between(range(len(x), len(x) + horizon), y_pred[:, 25, 0], y_pred[:, 75, 0], alpha=0.35, color="b",
#                      label="50% PI", linewidth=0.0)
#     plt.xlim([len(x) - offset, len(x) + horizon - 1])
#     plt.ylim([0, 1])
#     plt.xlabel('Lead time')
#     plt.ylabel('Power')
#     xticks = np.arange(len(x) - offset, len(x) + horizon, 4)
#     xtick_labels = np.arange(-offset, horizon, 4)
#     plt.xticks(xticks, xtick_labels)
#     plt.title(title)
#     plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
#     # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
#     plt.show()

def plot_evaluation(ncrps, nmae, period=24, title=''):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, period + 1), np.mean(ncrps, axis=(0, 2)), label='NCRPS')
    plt.plot(range(1, period + 1), np.mean(nmae, axis=(0, 2)), label='NMAE')
    plt.legend(fontsize=15)
    plt.xlabel('Lead time', fontsize=15)
    plt.grid()
    plt.xticks(range(1, period + 1), range(1, period + 1))
    plt.title(title)
    plt.show()
