import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# Methods for scaling features individually
# Fit the scalers
def fit_scalers(train: pd.DataFrame, scaler_dict) -> None:
    for name in train.columns:
        scaler_dict[name].fit(train[name].values.reshape(-1, 1))


# Apply scaling while preserving the data structure
def scale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame(scaler_dict[name].transform(data[name].values.reshape(-1, 1)),
                      index=data.index, columns=[name])
         for name in data.columns], axis=1)


# see scale
def unscale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame(scaler_dict[name].inverse_transform(data[name].values.reshape(-1, 1)),
                      index=data.index, columns=[name]) for name in data.columns], axis=1)


def preprocess_data(h_pars: dict):
    # h_pars must contain keys "horizon", "variables", "train_split"

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
    #ensembles.index = ensembles.index.droplevel(0)
    n_ens = len(ensembles.index.get_level_values(1).unique())

    # Split train and test set according to h_pars["train_split"]
    possible_dates = observations.index.map(lambda d: d.ceil(freq="D")).intersection( \
        observations.index.map(lambda d: d.floor(freq="D")))
    # round do even hour or stay if horizon =24
    dates = possible_dates.intersection(ensembles.index.get_level_values(0).unique().map(
        lambda d: d - pd.Timedelta(hours=h_pars["horizon"]))).map(
        lambda d: d + pd.Timedelta(hours=h_pars["horizon"]))
    n_obs = len(dates)
    n_train = int(len(dates) * h_pars["train_split"])
    # i_train = np.sort(np.random.choice(n_obs, size=n_train, replace=False))  # randomize the train and test set, not nice
    # i_test = np.delete(np.array(range(n_obs)), i_train)
    i_train = np.arange(0, n_train)
    i_test = np.arange(n_train, n_obs)
    dates_train = dates[i_train]
    dates_test = dates[i_test]
    # Select dates and add the wind power data to the weather ensembles
    ens_train = ensembles.loc[(dates_train, slice(None))]
    ens_train["wind_power"] = observations.loc[ens_train.index.get_level_values(0).map(
        lambda d: d - pd.Timedelta(hours=h_pars["horizon"]))].set_index(ens_train.index)
    ens_test = ensembles.loc[(dates_test, slice(None))]
    ens_test["wind_power"] = observations.loc[ens_test.index.get_level_values(0).map(
        lambda d: d - pd.Timedelta(hours=h_pars["horizon"]))].set_index(ens_test.index)
    # Here the power data has to be joined to the ensembles
    obs_train = observations.loc[dates_train]
    obs_test = observations.loc[dates_test]

    # Define scaler types for each variable
    scale_dict = {"u100": StandardScaler(), "v100": StandardScaler(), "t2m": StandardScaler(),
                  "sp": StandardScaler(),
                  "speed": MinMaxScaler(),
                  "wind_power": MinMaxScaler()}  # MinMaxScaler more suitable for power data.
    # But even better when not aggregated
    # Scale ensembles
    fit_scalers(ens_train, scale_dict)
    sc_ens_train = scale(ens_train, scale_dict)
    sc_ens_test = scale(ens_test, scale_dict)

    # Scale observations
    obs_scaler = scale_dict["wind_power"]
    # obs_scaler.fit(obs_train) # already scaled with the
    sc_obs_train = pd.DataFrame(data=obs_scaler.transform(obs_train), index=obs_train.index,
                                columns=obs_train.columns)
    sc_obs_test = pd.DataFrame(data=obs_scaler.transform(obs_test), index=obs_test.index,
                               columns=obs_test.columns)
    # Return the processed data
    return sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, scale_dict


# Reformat data depending on level of aggregation in h_pars["aggregation"]
def format_data(h_pars: dict, sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test):
    if h_pars["aggregation"] == "mean":
        sc_ens_train_f = sc_ens_train.groupby(level=0).agg(["mean"])
        sc_ens_test_f = sc_ens_test.groupby(level=0).agg(["mean"])
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    elif h_pars["aggregation"] == "mean+std":
        # mean the ensembles for each feature and add standard deviation
        sc_ens_train_f = sc_ens_train.groupby(level=0).agg(["mean", "std"])
        sc_ens_test_f = sc_ens_test.groupby(level=0).agg(["mean", "std"])
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    elif h_pars["aggregation"] == "single":
        sc_ens_train_f = sc_ens_train
        sc_ens_test_f = sc_ens_test
        # expand the index of sc_obs_train and _test and copy values relating to existing index levels
        sc_obs_train_f = pd.DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
        sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
            sc_obs_train["wind_power"]).values
        sc_obs_test_f = pd.DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
        sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
            sc_obs_test["wind_power"]).values
    elif h_pars["aggregation"] == "single+std":
        # use every ensemble member individually instead of mean of them -> more data
        # why does pandas not support addition of another level in a multiindex while copying values relating to the
        # existing levels?
        sc_ens_train_f = sc_ens_train.join(sc_ens_train.index.get_level_values(0).map(
            sc_ens_train.groupby(level=0).std().to_dict("index")).to_frame().set_index(
            sc_ens_train.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(
            labels=sc_ens_train.columns, axis=1),
            rsuffix="_std")
        sc_ens_test_f = sc_ens_test.join(sc_ens_test.index.get_level_values(0).map(
            sc_ens_test.groupby(level=0).std().to_dict("index")).to_frame().set_index(
            sc_ens_test.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(
            labels=sc_ens_test.columns, axis=1),
            rsuffix="_std")
        sc_obs_train_f = pd.DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
        sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
            sc_obs_train["wind_power"]).values
        sc_obs_test_f = pd.DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
        sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
            sc_obs_test["wind_power"]).values
    elif h_pars["aggregation"] == "all":  # give it all the info of the ensemble
        sc_ens_train_f = sc_ens_train[h_pars["variables"]].reset_index().pivot(index="time",
                                                                               columns="number")
        sc_ens_test_f = sc_ens_test[h_pars["variables"]].reset_index().pivot(index="time",
                                                                             columns="number")
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    else:
        raise Exception("Wrong aggregation method specified!")
    return sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f
