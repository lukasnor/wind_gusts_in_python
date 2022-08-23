import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.utils.np_utils import to_categorical


# Methods for scaling features individually
# Fit the scalers
def fit_scalers(train: DataFrame, scaler_dict) -> None:
    for name in scaler_dict:
        scaler_dict[name].fit(train[name].values.reshape(-1, 1))


# Apply scaling while preserving the data structure
# Empty scaler_dicts are allowed
def scale(data: DataFrame, scaler_dict) -> DataFrame:
    scaled_data = []
    for column in data.columns:
        if column in scaler_dict:
            scaled_np_column = scaler_dict[column].transform(data[column].values.reshape(-1, 1))  # the reshape is needed, otherwise scikit complains
            scaled_data.append(DataFrame(scaled_np_column, index=data.index, columns=[column]))
        else:
            scaled_data.append(data[column])
    return pd.concat(scaled_data, axis=1)


# see scale
def unscale(data: DataFrame, scaler_dict) -> DataFrame:
    scaled_data = []
    for column in data.columns:
        if column in scaler_dict:
            scaled_np_column = scaler_dict[column].inverse_transform(
                data[column].values.reshape(-1, 1))
            scaled_data.append(DataFrame(scaled_np_column, index=data.index, columns=[column]))
        else:
            scaled_data.append(data[column])
    return pd.concat(scaled_data, axis=1)


def obs_to_bins(obs: DataFrame, bin_edges: ndarray) -> DataFrame:
    helper_frame = pd.DataFrame(index=obs.index, columns=[*range(len(bin_edges))])
    helper_frame[[*range(len(bin_edges))]] = bin_edges
    return pd.concat([obs, helper_frame], axis=1) \
               .rank(axis=1) \
               .iloc[(slice(None), [0])] \
               .astype("int") - 2  # -2 to get the index of the bin, first bin being 0


def bins_to_categorical(binned_obs: DataFrame, num_classes) -> DataFrame:
    return DataFrame(index=binned_obs.index, data=to_categorical(binned_obs, num_classes))


def obs_to_categorical(obs: DataFrame, bin_edges: ndarray) -> DataFrame:
    # bin edges is one longer than n_bins, but we need one more bin in the end for bigger values
    return bins_to_categorical(obs_to_bins(obs, bin_edges), len(bin_edges))


# Manuell nen Validation set auswählen?
def import_data(horizon: int, variables: [str], train_split: float) -> (
        DataFrame, DataFrame, DataFrame, DataFrame):
    # Import observation data
    observations = pd.read_csv("../data/Sweden_Zone3_Power.csv", index_col=0)
    observations.index = pd.to_datetime(observations.index, infer_datetime_format=True)
    observations = observations[observations.columns.drop("horizon")]

    # Import ensemble data
    ensembles = pd.read_csv("../data/Sweden_Zone3_Ensembles.csv")
    ensembles["time"] = pd.to_datetime(ensembles["time"], infer_datetime_format=True)
    ensembles = ensembles.reset_index().pivot(index=["horizon", "time", "number"], columns=[])
    ensembles = ensembles[list(set(variables) & set(ensembles.columns))]
    # Select only relevant horizon
    ensembles = ensembles.sort_index(level=[0, 1, 2])
    ensembles = ensembles.loc[(horizon, slice(None), slice(None))]
    #ensembles.index = ensembles.index.droplevel(0)

    # Split train and test set according to h_pars["train_split"]
    possible_dates = observations.index.map(lambda d: d.ceil(freq="D")).intersection(
        observations.index.map(lambda d: d.floor(freq="D")))
    # round do even hour or stay if horizon =24
    dates = possible_dates.intersection(ensembles.index.get_level_values(0).unique().map(
        lambda d: d - pd.Timedelta(hours=horizon))).map(
        lambda d: d + pd.Timedelta(hours=horizon))
    n_obs = len(dates)
    n_train = int(len(dates) * train_split)
    # randomize the train and test set
    i_train = np.sort(np.random.choice(n_obs, size=n_train,
                                       replace=False))
    i_test = np.delete(np.array(range(n_obs)), i_train)
    # i_train = np.arange(0, n_train)  # split linearly
    # i_test = np.arange(n_train, n_obs)
    dates_train = dates[i_train]
    dates_test = dates[i_test]

    # Select dates and add the wind power data to the weather ensembles
    ens_train = ensembles.loc[(dates_train, slice(None))]
    ens_train["wind_power"] = observations.loc[ens_train.index.get_level_values(0).map(
        lambda d: d - pd.Timedelta(hours=horizon))].set_index(ens_train.index).astype("float64")
    ens_test = ensembles.loc[(dates_test, slice(None))]
    ens_test["wind_power"] = observations.loc[ens_test.index.get_level_values(0).map(
        lambda d: d - pd.Timedelta(hours=horizon))].set_index(ens_test.index).astype("float64")

    # Add dayofyear as sin and cos encoding
    ens_train["cos"] = ens_train.index.get_level_values(0).map(
        lambda timestamp: np.cos(2 * np.pi * (timestamp.dayofyear - 1) / 365))
    ens_train["sin"] = ens_train.index.get_level_values(0).map(
        lambda timestamp: np.sin(2 * np.pi * (timestamp.dayofyear - 1) / 365))
    ens_test["cos"] = ens_test.index.get_level_values(0).map(
        lambda timestamp: np.cos(2 * np.pi * (timestamp.dayofyear - 1) / 365))
    ens_test["sin"] = ens_test.index.get_level_values(0).map(
        lambda timestamp: np.sin(2 * np.pi * (timestamp.dayofyear - 1) / 365))

    obs_train = observations.loc[dates_train].astype("float64")
    obs_test = observations.loc[dates_test].astype("float64")

    return ens_train, ens_test, obs_train, obs_test


# scales input and output data according to input_variables and output_variables
# input_variables \subseteq ens_*.columns
# output_variable \subseteq obs_*.columns
# None variables means scaling all of them
# Empty lists mean scaling none of them
# TODO: Probeweise mal die Skalierung der Beobachtung rausnehmen
def scale_data(ens_train: DataFrame,
               ens_test: DataFrame,
               obs_train: DataFrame,
               obs_test: DataFrame,
               input_variables: [str] = None,
               output_variables: [str] = None) -> (
        DataFrame, DataFrame, DataFrame, DataFrame, dict, dict):
    # Define scaler types for each input variable
    input_scalers = {"u100": StandardScaler(),
                     "v100": StandardScaler(),
                     "t2m": StandardScaler(),
                     "sp": StandardScaler(),
                     "speed": MinMaxScaler(),
                     "wind_power": MinMaxScaler()}  # MinMaxScaler more suitable for power data.
    # But even better when not aggregated

    # Remove unnecessary scalers from the dict when variables are missing, not scaling them
    if input_variables is not None:
        input_scalers = {variable: input_scalers[variable] for variable in input_variables}

    # Scale ensembles
    fit_scalers(ens_train, input_scalers)
    sc_ens_train = scale(ens_train, input_scalers)
    sc_ens_test = scale(ens_test, input_scalers)

    # Same for the observations
    output_scalers = {"wind_power": MinMaxScaler()}
    if output_variables is not None:
        output_scalers = {variable: output_scalers[variable] for variable in output_variables}
    fit_scalers(obs_train, output_scalers)
    sc_obs_train = scale(obs_train, output_scalers)
    sc_obs_test = scale(obs_test, output_scalers)

    # Return the processed data
    return sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers


# First import, then scale
def preprocess_data(horizon: int,
                    train_variables: [str],
                    train_split: float,
                    input_variables: [str] = None,
                    output_variables: [str] = None) -> (
        DataFrame, DataFrame, DataFrame, DataFrame, dict, dict):
    return scale_data(*import_data(horizon, train_variables, train_split), input_variables,
                      output_variables)


# Reformat data depending on level of aggregation in h_pars["aggregation"]
def format_data(sc_ens_train: DataFrame,
                sc_ens_test: DataFrame,
                sc_obs_train: DataFrame,
                sc_obs_test: DataFrame,
                aggregation: str) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    constant_variables = pd.Index(["wind_power", "cos", "sin"])
    nonconstant_variables = sc_ens_train.columns.drop(constant_variables)
    if aggregation == "mean":
        sc_ens_train_f = sc_ens_train.groupby(level=0).agg(["mean"])
        sc_ens_test_f = sc_ens_test.groupby(level=0).agg(["mean"])
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    elif aggregation == "mean+std":
        # mean the ensembles for each feature and add standard deviation
        sc_ens_train_f = sc_ens_train[nonconstant_variables].groupby(level=0).agg(["mean", "std"])
        sc_ens_test_f = sc_ens_test[nonconstant_variables].groupby(level=0).agg(["mean", "std"])
        sc_ens_train_f = pd.concat(
            [sc_ens_train_f, sc_ens_train[constant_variables].groupby(level=0).agg(["mean"])],
            axis=1)
        sc_ens_test_f = pd.concat(
            [sc_ens_test_f, sc_ens_test[constant_variables].groupby(level=0).agg("mean")], axis=1)
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    elif aggregation == "single":
        sc_ens_train_f = sc_ens_train
        sc_ens_test_f = sc_ens_test
        # expand the index of sc_obs_train and _test and copy values relating to existing index levels
        sc_obs_train_f = DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
        sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
            sc_obs_train["wind_power"]).values
        sc_obs_test_f = DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
        sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
            sc_obs_test["wind_power"]).values
    elif aggregation == "single+std":
        # use every ensemble member individually instead of mean of them -> more data
        # split constant and nonconstant variables, format only the nonconstant ones
        constant_train = sc_ens_train[constant_variables]
        constant_test = sc_ens_test[constant_variables]
        sc_ens_train = sc_ens_train[nonconstant_variables]
        sc_ens_test = sc_ens_test[nonconstant_variables]
        # why does pandas not support addition of another level in a multiindex while copying values relating to the
        # existing levels?
        sc_ens_train_f = sc_ens_train.join(sc_ens_train.index.get_level_values(0).map(
            sc_ens_train.groupby(level=0).std().to_dict("index")).to_frame().set_index(
            sc_ens_train.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(
            labels=sc_ens_train.columns, axis=1),
            rsuffix="_std")
        sc_ens_train_f = pd.concat([sc_ens_train_f, constant_train], axis=1)
        sc_ens_test_f = sc_ens_test.join(sc_ens_test.index.get_level_values(0).map(
            sc_ens_test.groupby(level=0).std().to_dict("index")).to_frame().set_index(
            sc_ens_test.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(
            labels=sc_ens_test.columns, axis=1),
            rsuffix="_std")
        sc_ens_test_f = pd.concat([sc_ens_test_f, constant_test], axis=1)
        sc_obs_train_f = DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
        sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
            sc_obs_train["wind_power"]).values
        sc_obs_test_f = DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
        sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
            sc_obs_test["wind_power"]).values
    elif aggregation == "all":  # give it all the info of the ensemble
        # Since the wind_power #horizon ahead is the same for each ensemble member, make it only one
        # column
        sc_ens_train_f = pd.concat([sc_ens_train.reset_index().pivot(index="time",
                                                                     columns="number",
                                                                     values=sc_ens_train.columns.drop(
                                                                         constant_variables)
                                                                     ),
                                    sc_ens_train[constant_variables].groupby(level=0).agg(
                                        ["mean"])],
                                   axis=1)
        sc_ens_test_f = pd.concat([sc_ens_test.reset_index().pivot(index="time",
                                                                   columns="number",
                                                                   values=sc_ens_test.columns.drop(
                                                                       constant_variables)
                                                                   ),
                                   sc_ens_test[constant_variables].groupby(level=0).agg(["mean"])],
                                  axis=1)
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    else:
        raise Exception("Wrong aggregation method specified!")
    return sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f


# Add the categorical data to the observations and inputs
def categorify_data(sc_ens_train_f: DataFrame,
                    sc_ens_test_f: DataFrame,
                    sc_obs_train_f: DataFrame,
                    sc_obs_test_f: DataFrame,
                    bin_edges: np.ndarray) -> (DataFrame, DataFrame, DataFrame, DataFrame):
    # For the training data, drop the last category, since no observation is outside the last bin
    cat_ens_train = obs_to_categorical(sc_ens_train_f[["wind_power"]], bin_edges).iloc[:, :-1]
    cat_obs_train = obs_to_categorical(sc_obs_train_f[["wind_power"]], bin_edges).iloc[:, :-1]
    sc_ens_train_fc = pd.concat([sc_ens_train_f, cat_ens_train], axis=1)
    sc_obs_train_fc = pd.concat([sc_obs_train_f, cat_obs_train], axis=1)

    # For the test data, merge all the data bigger than the last bin_edge into the last bin
    cat_ens_test = obs_to_categorical(sc_ens_test_f[["wind_power"]], bin_edges)
    cat_ens_test.iloc[:, -2] = cat_ens_test.iloc[:, -2:].sum(axis=1)
    cat_ens_test = cat_ens_test.iloc[:, :-1]
    cat_obs_test = obs_to_categorical(sc_obs_test_f[["wind_power"]], bin_edges)
    cat_obs_test.iloc[:, -2] = cat_obs_test.iloc[:, -2:].sum(axis=1)
    cat_obs_test = cat_obs_test.iloc[:, :-1]
    sc_ens_test_fc = pd.concat([sc_ens_test_f, cat_ens_test], axis=1)
    sc_obs_test_fc = pd.concat([sc_obs_test_f, cat_obs_test], axis=1)

    return sc_ens_train_fc, sc_ens_test_fc, sc_obs_train_fc, sc_obs_test_fc


if __name__ == "__main__":
    h_pars = {"horizon": 3,
              "variables": ["t2m", "sp", "wind_power"],
              "train_split": 0.85,

              "aggregation": "all"}
    ens_train, ens_test, obs_train, obs_test = import_data(h_pars["horizon"],
                                                           h_pars["variables"],
                                                           h_pars["train_split"])
    sc_ens_train, \
    sc_ens_test, \
    sc_obs_train, \
    sc_obs_test, \
    input_scalers, \
    output_scalers = scale_data(ens_train,
                                ens_test,
                                obs_train,
                                obs_test,
                                h_pars["variables"])
    sc_ens_train_f, \
    sc_ens_test_f, \
    sc_obs_train_f, \
    sc_obs_test_f = format_data(sc_ens_train,
                                sc_ens_test,
                                sc_obs_train,
                                sc_obs_test, h_pars["aggregation"])
