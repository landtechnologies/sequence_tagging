from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    # model.restore_session("results/crf/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, max_iter=config.max_iter)
    train = CoNLLDataset(config.filename_train, max_iter=config.max_iter)

    # train model
    model.train(train, dev)

if __name__ == "__main__":
    main()
