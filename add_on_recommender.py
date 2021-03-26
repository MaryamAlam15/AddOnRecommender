import pandas as pd

from config import AddOnProductsConfig
from train_model import TrainModel

pd.set_option("display.max_rows", 500, "display.max_columns", 100)
pd.set_option('display.max_colwidth', 500)
pd.options.mode.chained_assignment = None


class AddOnRecommender:

    def __init__(self):
        self.config = AddOnProductsConfig()

    def run(self):
        TrainModel(self.config).train()


if __name__ == '__main__':
    AddOnRecommender().run()
