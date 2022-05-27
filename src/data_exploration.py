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


def plot_wind_power_data_split(plots_path=None):
    h_pars = {"horizon": 3, "variables": ["sp"], "train_split": 0.85}
    # Import observation data
    observations = pd.read_csv("../data/Sweden_Zone3_Power.csv", index_col=0)
    observations.index = pd.to_datetime(observations.index, infer_datetime_format=True)
    observations = observations[observations.columns.drop("horizon")]

    # Import ensemble data
    ensembles = pd.read_csv("../data/Sweden_Zone3_Ensembles.csv")
    ensembles["time"] = pd.to_datetime(ensembles["time"], infer_datetime_format=True)
    ensembles = ensembles.reset_index().pivot(index=["horizon", "time", "number"], columns=[])
    # variables = ensembles.columns.drop(["is_origin", "index"])  # Not needed anymore
    ensembles = ensembles[h_pars["variables"]]
    # Select only relevant horizon
    ensembles = ensembles.sort_index(level=[0, 1, 2])
    ensembles = ensembles.loc[(h_pars["horizon"], slice(None), slice(None))]
    ensembles.index = ensembles.index.droplevel(0)
    # Split train and test set according to h_pars["train_split"]
    possible_dates = observations.index.map(lambda d: d.ceil(freq="D")).intersection( \
        observations.index.map(lambda d: d.floor(freq="D")))
    # round do even hour or stay if horizon =24
    dates = possible_dates.intersection(ensembles.index.get_level_values(0).unique().map(
        lambda d: d - pd.Timedelta(hours=h_pars["horizon"]))).map(
        lambda d: d + pd.Timedelta(hours=h_pars["horizon"]))
    n_obs = len(dates)
    n_train = int(len(dates) * h_pars["train_split"])
    i_train = np.arange(0, n_train)
    i_test = np.arange(n_train, n_obs)
    dates_train = dates[i_train]
    dates_test = dates[i_test]
    obs_train = observations.loc[dates_train]
    obs_test = observations.loc[dates_test]
    with plt.xkcd():
        obs_train.hist(figsize=figsize, bins=50, grid=False)
        plt.title("Wind Power Histogram - Train", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path+"wind_power_histogram_train.png")
        obs_test.hist(figsize=figsize, bins=50, grid=False)
        plt.title("Wind Power Histogram - Test", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path + "wind_power_histogram_test.png")

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
    #plot_wind_power_data(plots_path="../results/data analysis/wind_power_histogram.png")
    plot_wind_power_data_split(plots_path="../results/data analysis/")
