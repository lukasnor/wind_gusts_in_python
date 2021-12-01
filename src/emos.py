import datetime

import pandas as pd
import pywatts.modules
from pywatts.core.pipeline import Pipeline
from EmosModel import EmosModel


def create_preprocessing_pipeline(modules):
    pipeline = Pipeline(path="results/preprocessing")

    return pipeline


def create_training_pipeline(modules):
    pipeline = Pipeline(path="results/training")

    return pipeline


if __name__ == "__main__":
    train = pd.read_csv("data/df_train.csv", index_col=0)
    test = pd.read_csv("data/df_test.csv", index_col=0)

    loc2_train = train[train["location"] == 2]
    loc2_test = test[test["location"] == 2]

    # Make the first dimension to DateTimeIndex, else Pipeline.transform() breaks
    start = datetime.datetime(2015,1,1,0,0)
    index = pd.date_range(start, periods=len(loc2_train), freq="H")
    loc2_train.set_index(index, drop=True,append=False,inplace=True)

    pipeline = Pipeline(path="results/")
    emos_model = EmosModel(name="Emos")
    test_model = emos_model(obs=pipeline["obs"], ens_mean=pipeline["ens_mean"], ens_sd=pipeline["ens_sd"]
                            # , computation_mode=pywatts.core.computation_mode.ComputationMode(2)
                            )

    pipeline.train(data=loc2_train)
