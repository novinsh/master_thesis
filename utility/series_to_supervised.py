import pandas as pd
# Adapted from machinelearningmastery.com


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True, split=False):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
        split: whether to split into instances and targets or not.
    Returns:
        Pandas DataFrame of series framed for supervised learning + the variable names mapping
    """
    if len(data.shape) == 1 or type(data) is list:
        n_vars = 1
    else:
        n_vars = data.shape[1]
    # TODO: useful for multivariate cases
    # var_mapping = {data.columns.values[j]: 'var%d' % (j + 1) for j in range(n_vars)}
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # split the training instances and its targets
    if split:
        x_df = pd.DataFrame()
        y_df = pd.DataFrame()
        for i in range(1, n_vars + 1):
            col_select = [col for col in agg.columns if col.startswith('var{0}'.format(i))]
            df = agg.loc[:, col_select]
            x, y = df.iloc[:, :n_in], df.iloc[:, -n_out:]
            x_df = pd.concat((x_df, x), axis=1)
            y_df = pd.concat((y_df, y), axis=1)
        return x_df, y_df
    else:
        return agg


if __name__ == "__main__":
    df_power = pd.read_pickle("../data/df_power.pkl")

    # univariate time series case (for showcase)
    n_input = 2
    n_output = 2
    n_size = 10
    data = series_to_supervised(df_power.wf1, n_in=n_input, n_out=n_output)
    X, y = data.iloc[:n_size, :n_input], data.iloc[:n_size, -n_output:]
    print(X.shape)
    print(y.shape)
    #
    from matplotlib import pyplot as plt
    plt.plot(X, '.-')
    plt.plot(y, '.-')
    plt.grid()
    train_labels = ['$y_{t} (train)$']
    train_labels += ['$y_{t-%i} (train)$' % (i) for i in range(1, X.shape[1])]
    test_labels = ['$y_{t+%i} (label)$' % (i + 1) for i in range(y.shape[1])]
    plt.legend(train_labels + test_labels)
    plt.show()

    # test the split
    print("test split")
    X, y = series_to_supervised(df_power.wf1, n_in=n_input, n_out=n_output, split=True)
    print(X.shape)
    print(X.columns.values)
    print(y.shape)
    print(y.columns.values)

    # multivariate time serie case
    # TODO: demonstrate, perhaps will be also useful for plots in the thesis later!
