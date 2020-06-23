import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
import keras.backend as K


class CustomGen(TimeseriesGenerator):
    def __getitem__(self, index):
        if self.shuffle:
            rows = np.random.randint(
                self.start_index, self.end_index + 1, size=self.batch_size)
        else:
            i = self.start_index + self.batch_size * self.stride * index
            rows = np.arange(i, min(i + self.batch_size *
                                    self.stride, self.end_index + 1), self.stride)

        samples = np.array([self.data[row - self.length:row:self.sampling_rate]
                            for row in rows])
        targets = np.array([self.targets[row] for row in rows])

        if self.reverse:
            return samples[:, ::-1, ...], targets
        return samples, [targets,targets]


def mean_loss(log_var):
    def customLoss(yTrue, yPred):
        loss1 = K.mean(K.exp(-log_var) * K.square(yTrue - yPred))
        return loss1
    return customLoss


def var_loss(log_var):
    def customLoss(yTrue, yPred):
        #loss2 = K.sqrt(K.exp(log_var))
        loss2 = K.mean(log_var) # why this works and above doesn't?!
        return loss2
    return customLoss
