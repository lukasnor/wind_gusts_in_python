import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 24}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}
figsize = (13, 7)


def plot_wind_power_data(plots_path=None):
    wind_power = pd.read_csv("../data/Sweden_Zone3_Power.csv", index_col="time")
    wind_power = wind_power["wind_power"]
    with plt.xkcd(scale=0.1, length=0.1, randomness=0.1):
        wind_power.hist(figsize=figsize, bins=100, grid=False)
        plt.title("Wind Power Histogram", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path)


def whatever():
    train_data = pd.read_csv("data/df_train.csv", index_col=0)
    # test_data = pd.read_csv("data/df_test.csv", index_col=0)

    # train_data.info()

    loc1 = train_data[train_data["location"] == 1]  # only data from location 1
    # train_data.groupby("location")["ens_mean"].mean()
    # loc1.info()
    # loc1.iloc[:,7:27].plot()
    # plt.show()
    test_data = pd.DataFrame({"loc": loc1.iloc[0], "obs": loc1.iloc[1]})
    test_data.info()
    npa = np.array(loc1.iloc[1])
    print(npa.mean())


if __name__ == "__main__":
    plot_wind_power_data(plots_path="../results/data analysis/wind_power_histogram.png")
