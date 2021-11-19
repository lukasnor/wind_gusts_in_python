import pandas as pd

if __name__ == "__main__":
    train_data = pd.read_csv("../data/df_train.csv", index_col=0)
    test_data = pd.read_csv("../data/df_test.csv", index_col=0)

    loc1 = train_data[train_data["location"] == 1]  # only data from location 1
    train_data.groupby("location")["ens_mean"].mean()
