import pandas as pd
from pywatts.core.pipeline import Pipeline

if __name__ == "__main__":

    # Importing the data
    data_train = pd.read_csv("../data/df_train.csv")
    data_test = pd.read_csv("../data/df_test.csv")



    pipeline = Pipeline(path="../results")
    # pipeline.train(data)

    print("FINISHED", end="")
