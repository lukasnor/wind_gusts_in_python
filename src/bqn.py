import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from scipy.special import binom
from keras import Model, layers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

fontdict_title = {"fontweight": "bold", "fontsize": 18}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}

h_pars = {"degree": 10,
          "layer_sizes": [16, 16, 16],
          "activations": None,
          "batch_size": 32,
          "patience": 20,
          "variables": None,
          "horizon": 6,
          "aggregation": "single"  # possible implementations: "mean", "single", "single+variance", "all"
          }
# Default value for activation is "selu" if activations do not match layer_sizes
if h_pars["activations"] is None or \
        not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
    h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]


# Construction of the local model
def get_local_model(name: str, n_input: int, n_loc: int, n_emb: int, layer_sizes: [int], activations: [str],
                    degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=n_input, name="ens_input")
    emb_input = layers.Input(shape=1, name="emb_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc, output_dim=n_emb, input_shape=1)(emb_input)
    station_embedding_flat = layers.Flatten()(station_embedding_part)
    # Merge Inputs
    merged = layers.Concatenate(name="merged")([ens_input, station_embedding_flat])

    # Hidden layers
    hidden1 = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden1")(merged)
    hidden2 = layers.Dense(units=layer_sizes[1], activation=activations[1], name="hidden2")(hidden1)

    # Output
    output = layers.Dense(units=degree + 1, activation="softplus")(hidden2)  # smooth, non-negative and proportional

    # Model
    return Model(name=name, inputs=[ens_input, emb_input], outputs=output)


# Construction of model
def get_model(name: str, n_input: int, layer_sizes: [int], activations: [str],
              degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=n_input, name="ens_input")

    # Hidden layers
    x = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden0")(ens_input)
    for i in range(len(layer_sizes) - 1):
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1], name="hidden" + str(i + 1))(x)

    # Output
    output = layers.Dense(units=degree + 1, activation="softplus")(x)  # smooth, non-negative and proportional

    # Model
    return Model(name=name, inputs=ens_input, outputs=output)


# Construction of the loss function
degree = h_pars["degree"]
loss_quantile_levels = np.arange(0.01, 1, 0.01)  # 1% to 99% quantile levels for the loss, equidistant
lql = tf.constant(loss_quantile_levels, dtype="float32")
bernsteins = np.array(
    [binom(degree, j) * np.power(loss_quantile_levels, j) * np.power(1 - loss_quantile_levels, degree - j) for j in
     range(degree + 1)]).transpose()
B = tf.constant(bernsteins, dtype="float32")


# Multi-quantile loss: mean over all quantile losses for levels in lql
def quantile_loss(y_true, y_pred):
    quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
    error = y_true - quantiles  # no expand dims
    err1 = error * tf.expand_dims(lql - 1, 0)
    err2 = error * tf.expand_dims(lql, 0)
    loss = tf.maximum(err1, err2)
    return tf.reduce_sum(loss, axis=1)


# Methods for scaling features individually
# Fit the scalers
def fit_scalers(train: pd.DataFrame, scaler_dict) -> None:
    for name, scaler in scaler_dict.items():
        scaler.fit(train[name].values.reshape(-1, 1))


# Apply scaling while preserving the data structure
def scale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame(scaler.transform(data[name].values.reshape(-1, 1)), index=data.index, columns=[name]) for
         name, scaler in scaler_dict.items()], axis=1)


# see scale
def unscale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame(scaler.inverse_transform(data[name].values.reshape(-1, 1)), index=data.index, columns=[name]) for
         name, scaler in scaler_dict.items()], axis=1)


# From the model prediction, i.e. the increments of the coefficients of bernstein pols, calculate the actual quantiles
def get_quantiles(y_pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]])),
                        index=y_pred.index)


# Calculate the rank of the observation in the quantile forecast
def get_rank(obs: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([obs, quantiles], axis=1).rank(axis=1).iloc[:, 0]


def generate_pit_plot(obs: pd.DataFrame, quantiles: pd.DataFrame, name: str,
                      n_bins: int = len(loss_quantile_levels) + 1) -> None:
    ranks = get_rank(obs, quantiles)
    plt.hist(ranks, bins=n_bins)
    plt.hlines(len(ranks) / n_bins, linestyles="dashed", color="black", xmin=0, xmax=100)
    plt.title("Rank Histogram - " + name, fontdict=fontdict_title)
    plt.show()


def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, n=None) -> None:
    q = get_quantiles(y_pred)
    if n is None:
        n = y_true.shape[0]
    for i in range(n):
        plt.plot(q.iloc[(i, slice(None))], loss_quantile_levels, color="blue", label="forecast")
        plt.vlines(y_true.iloc[i], ymin=loss_quantile_levels.min(), ymax=loss_quantile_levels.max(),
                   label="observation", color="red", linestyles="dashed")
        plt.xlim(left=0.0, right=max(1.0, plt.axis()[1]))
        plt.title("Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        plt.show()


# Define forecast horizons
horizons = [3, 6, 9, 12, 15, 18, 21, 24]
horizon = h_pars["horizon"]

# Import observation data
observations = pd.read_csv("../data/Sweden_Zone3_Power.csv", index_col=0)
observations.index = pd.to_datetime(observations.index, infer_datetime_format=True)
observations = observations[observations.columns.drop("horizon")]

# Import ensemble data
ensembles = pd.read_csv("../data/Sweden_Zone3_Ensembles.csv")
ensembles["time"] = pd.to_datetime(ensembles["time"], infer_datetime_format=True)
ensembles = ensembles.reset_index().pivot(index=["horizon", "time", "number"], columns=[])
variables = ensembles.columns.drop(["is_origin", "index"])
if h_pars["variables"] is not None:
    variables = h_pars["variables"]
ensembles = ensembles[variables]
# Select only relevant horizon
ensembles = ensembles.sort_index(level=[0, 1, 2])
ensembles = ensembles.loc[(horizon, slice(None), slice(None))]
ensembles.index = ensembles.index.droplevel(0)
n_ens = len(ensembles.index.get_level_values(1).unique())

# Split train and test set
train_split = 0.85
dates = observations.index.intersection(ensembles.index.get_level_values(0).unique())
n_obs = len(dates)
n_train = int(len(dates) * train_split)
# i_train = np.sort(np.random.choice(n_obs, size=n_train, replace=False))  # randomize the train and test set, not nice
# i_test = np.delete(np.array(range(n_obs)), i_train)
i_train = np.arange(0, n_train)
i_test = np.arange(n_train, n_obs)
dates_train = dates[i_train]
dates_test = dates[i_test]
ens_train = ensembles.loc[(dates_train, slice(None))]
ens_test = ensembles.loc[(dates_test, slice(None))]
obs_train = observations.loc[dates_train]
obs_test = observations.loc[dates_test]

# Define scaler types for each variable
scale_dict = {"u100": StandardScaler(), "v100": StandardScaler(), "t2m": StandardScaler(), "sp": StandardScaler(),
              "speed": MinMaxScaler()}
# Scale ensembles
fit_scalers(ens_train, scale_dict)
sc_ens_train = scale(ens_train, scale_dict)
sc_ens_test = scale(ens_test, scale_dict)

# Scale observations
obs_scaler = MinMaxScaler()  # MinMaxScaler more suitable for power data. But even better when not aggregated
obs_scaler.fit(obs_train)
sc_obs_train = pd.DataFrame(data=obs_scaler.transform(obs_train), index=obs_train.index, columns=obs_train.columns)
sc_obs_test = pd.DataFrame(data=obs_scaler.transform(obs_test), index=obs_test.index, columns=obs_test.columns)

# Reformat data depending on level of aggregation
if h_pars["aggregation"] == "mean":  # mean the ensembles for each feature
    sc_ens_train_f = sc_ens_train.groupby(level=0).mean()
    sc_ens_test_f = sc_ens_test.groupby(level=0).mean()
    sc_obs_train_f = sc_obs_train
    sc_obs_test_f = sc_obs_test
if h_pars["aggregation"] == "single":  # use every ensemble member individually instead of mean of them -> more data
    sc_ens_train_f = sc_ens_train
    sc_ens_test_f = sc_ens_test
    sc_obs_train_f = pd.DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
    sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
        sc_obs_train["wind_power"]).values
    sc_obs_test_f = pd.DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
    sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
        sc_obs_test["wind_power"]).values
if h_pars["aggregation"] in ["mean", "single"]:  # If not all ensemble members are used, pass through variance by hand
    None  # TODO: Add variances to data
if h_pars["aggregation"] == "all":  # give it all the info of the ensemble
    sc_ens_train_f = sc_ens_train[variables].reset_index().pivot(index="time", columns="number")
    sc_ens_test_f = sc_ens_test[variables].reset_index().pivot(index="time", columns="number")
    sc_obs_train_f = sc_obs_train
    sc_obs_test_f = sc_obs_test

# Compile model
model = get_model(name="Nouny", n_input=len(sc_ens_train_f.columns), layer_sizes=h_pars["layer_sizes"],
                  activations=h_pars["activations"], degree=h_pars["degree"])
model.compile(optimizer="adam", loss=quantile_loss)

# Fit model
history = model.fit(y=sc_obs_train_f,
                    x=sc_ens_train_f,
                    batch_size=h_pars["batch_size"],
                    epochs=500,
                    verbose=1,
                    validation_freq=1,
                    validation_split=0.20,
                    callbacks=[EarlyStopping(patience=h_pars["patience"], restore_best_weights=True)],
                    use_multiprocessing=True
                    )

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.xlabel("Epochs", fontdict=fontdict_axis)
plt.title("Training Plot - Horizon " + str(horizon), fontdict=fontdict_title)
plt.show()

# Evaluate model
train = pd.DataFrame(model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
generate_pit_plot(sc_obs_train_f, get_quantiles(train), "Training set - Horizon " + str(horizon), n_bins=50)
test = pd.DataFrame(model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
generate_forecast_plots(sc_obs_test_f[::51], test[::51], n=20)
generate_pit_plot(sc_obs_test_f, get_quantiles(test), "Test set - Horizon " + str(horizon), n_bins=50)
