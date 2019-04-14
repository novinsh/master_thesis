from models.Model import Model


class Bare(Model):
    def architecture(self):
        print("To implement the model architecture")

    def fit(self, train, config):
        self.architecture()
        print("To implement training")

    def predict(self, test):
        print("To implement prediction")


if __name__ == "__main__":
    bare_config = {
        'n_input': 1,
        'n_output': 1,
        'n_epochs': 1000,
        'n_batch': 32,
        'lr': 0.01,
        'n_bins': 10
    }

    bare = Bare(bare_config)
    bare.fit(None,bare_config)
    bare.predict(None)
    bare.pred_2_rv([1,2,3])

