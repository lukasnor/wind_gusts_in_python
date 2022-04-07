import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers
from keras.callbacks import EarlyStopping
from scipy.special import binom
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 18}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}


# Construction of the local model
def get_local_model(name: str, input_size: int, n_loc: int, n_emb: int, layer_sizes: [int], activations: [str],
                    degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=input_size, name="ens_input")
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
def get_model(name: str, input_size: int, layer_sizes: [int], activations: [str],
              degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=input_size, name="ens_input")

    # Hidden layers
    x = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden0")(ens_input)
    for i in range(len(layer_sizes) - 1):
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1], name="hidden" + str(i + 1))(x)

    # Output
    output = layers.Dense(name="output", units=degree + 1, activation="softplus")(
        x)  # smooth, non-negative and proportional

    # Model
    return Model(name=name, inputs=ens_input, outputs=output)


# Use average_models instead! As get_model, but averaged over #trials
def get_averaged_model(name: str, n_input: int, layer_sizes: [int], activations: [str],
                       degree: int, trials: int) -> Model:
    common_input = layers.Input(name="common_input", shape=n_input)
    quantile_loss = build_quantile_loss(degree)
    models = [get_model("submodel" + str(i), n_input, layer_sizes, activations, degree) for i
              in range(trials)]
    for model in models:
        model.compile(loss=quantile_loss, optimizer="adam")
    print(models)
    print(models[0].summary())
    avg_output = layers.Average(name="average")([model(common_input) for model in models])
    return Model(name=name, inputs=common_input, outputs=avg_output)


# For a list of models of same type, construct the average model
def average_models(models: [Model]) -> Model:
    common_input = layers.Input(name="common_input", shape=models[0].input.shape[1])
    average_output = layers.Average(name="average")([model(common_input) for model in models])
    return Model(name="average_model", inputs=common_input, outputs=average_output)


# Construction of the loss function
def build_quantile_loss(degree: int):  # -> Loss function
    loss_quantile_levels = np.arange(0.01, 1, 0.01)
    lql = tf.constant(loss_quantile_levels, dtype="float32")  # 1% to 99% quantile levels for the loss, equidistant
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(loss_quantile_levels, j) * np.power(1 - loss_quantile_levels, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")  # Bernstein polynomials to interpolate the CDF

    # Multi-quantile loss: sum over all quantile losses for levels in lql
    def quantile_loss(y_true, y_pred):
        quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = y_true - quantiles  # no expand dims
        err1 = error * tf.expand_dims(lql - 1, 0)
        err2 = error * tf.expand_dims(lql, 0)
        loss = tf.maximum(err1, err2)
        return tf.reduce_sum(loss, axis=1)

    return quantile_loss


# Methods for scaling features individually
# Fit the scalers
def fit_scalers(train: pd.DataFrame, scaler_dict) -> None:
    for name in train.columns:
        scaler_dict[name].fit(train[name].values.reshape(-1, 1))


# Apply scaling while preserving the data structure
def scale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat(
        [pd.DataFrame(scaler_dict[name].transform(data[name].values.reshape(-1, 1)), index=data.index, columns=[name])
         for name in data.columns], axis=1)


# see scale
def unscale(data: pd.DataFrame, scaler_dict) -> pd.DataFrame:
    return pd.concat([pd.DataFrame(scaler_dict[name].inverse_transform(data[name].values.reshape(-1, 1)),
                                   index=data.index, columns=[name]) for name in data.columns], axis=1)


# From the model prediction, i.e. the increments of the coefficients of bernstein pols, calculate the actual quantiles
def get_quantiles(y_pred: pd.DataFrame, quantile_levels: np.ndarray) -> pd.DataFrame:
    degree = len(y_pred.columns) - 1
    bernsteins = np.array(
        [binom(degree, j) * np.power(quantile_levels, j) * np.power(1 - quantile_levels, degree - j) for j in
         range(degree + 1)]).transpose()
    return pd.DataFrame(np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]])),
                        index=y_pred.index)


# Calculate the rank of the observation in the quantile forecast
def get_rank(obs: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([obs, quantiles], axis=1).rank(axis=1).iloc[:, 0]


def generate_pit_plot(obs: pd.DataFrame, quantiles: pd.DataFrame, name: str,
                      n_bins: int) -> None:
    ranks = get_rank(obs, quantiles)
    plt.hist(ranks, bins=n_bins)
    plt.hlines(len(ranks) / n_bins, linestyles="dashed", color="black", xmin=0, xmax=100)
    plt.title("Rank Histogram - " + name, fontdict=fontdict_title)
    plt.show()


def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, quantile_levels: np.ndarray, n=None) -> None:
    q = get_quantiles(y_pred, quantile_levels=quantile_levels)
    if n is None:
        n = y_true.shape[0]
    for i in range(n):
        plt.plot(q.iloc[(i, slice(None))], quantile_levels, color="blue", label="forecast")
        plt.vlines(y_true.iloc[i], ymin=quantile_levels.min(), ymax=quantile_levels.max(),
                   label="observation", color="red", linestyles="dashed")
        plt.xlim(left=0.0, right=max(1.0, plt.axis()[1]))
        plt.title("Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        plt.show()


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
    dates = observations.index.intersection(ensembles.index.get_level_values(0).unique())
    n_obs = len(dates)
    n_train = int(len(dates) * h_pars["train_split"])
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

    # Return the processed data
    return sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test


# Reformat data depending on level of aggregation in h_pars["aggregation"]
def format_data(h_pars: dict, sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test):
    if h_pars["aggregation"] == "mean":  # mean the ensembles for each feature
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
            sc_ens_train.groupby(level=0).std().to_dict("index")).to_frame().set_index(sc_ens_train.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(labels=sc_ens_train.columns, axis=1),
                                           rsuffix="_std")
        sc_ens_test_f = sc_ens_test.join(sc_ens_test.index.get_level_values(0).map(
            sc_ens_test.groupby(level=0).std().to_dict("index")).to_frame().set_index(sc_ens_test.index).apply(
            func=lambda d: d[0].values(), axis=1, result_type="expand").set_axis(labels=sc_ens_test.columns, axis=1),
                                         rsuffix="_std")
        sc_obs_train_f = pd.DataFrame(index=sc_ens_train.index, columns=sc_obs_train.columns)
        sc_obs_train_f["wind_power"] = pd.Series(sc_obs_train_f.index.get_level_values(0)).map(
            sc_obs_train["wind_power"]).values
        sc_obs_test_f = pd.DataFrame(index=sc_ens_test.index, columns=sc_obs_test.columns)
        sc_obs_test_f["wind_power"] = pd.Series(sc_obs_test_f.index.get_level_values(0)).map(
            sc_obs_test["wind_power"]).values
    elif h_pars["aggregation"] == "all":  # give it all the info of the ensemble
        sc_ens_train_f = sc_ens_train[h_pars["variables"]].reset_index().pivot(index="time", columns="number")
        sc_ens_test_f = sc_ens_test[h_pars["variables"]].reset_index().pivot(index="time", columns="number")
        sc_obs_train_f = sc_obs_train
        sc_obs_test_f = sc_obs_test
    else:
        raise Exception("Wrong aggregation method specified!")
    return sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f


if __name__ == "__main__":
    # Get the data in a processed form
    h_pars = {"horizon": 6,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "single+std",
              "degree": 16,
              "layer_sizes": [16, 16, 20],
              "activations": ["selu", "selu", "selu"],

              "batch_size": 100,
              "patience": 50,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed"]

    # Import the data
    sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test = preprocess_data(h_pars=h_pars)

    # Format the data
    sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f = format_data(h_pars, sc_ens_train, sc_ens_test,
                                                                               sc_obs_train, sc_obs_test)

    # Average over models
    models = []
    for i in range(1):
        # Compile model
        model = get_model(name="Nouny" + str(i), input_size=len(sc_ens_train_f.columns), layer_sizes=h_pars["layer_sizes"],
                          activations=h_pars["activations"], degree=h_pars["degree"])
        model.compile(optimizer="adam", loss=build_quantile_loss(h_pars["degree"]))

        # Fit model
        history = model.fit(y=sc_obs_train_f,
                            x=sc_ens_train_f,
                            batch_size=h_pars["batch_size"],
                            epochs=500,
                            verbose=1,
                            validation_freq=1,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(patience=h_pars["patience"]
                                                     # , restore_best_weights=True
                                                     )],
                            use_multiprocessing=True
                            )
        # Plot the learning curves
        plt.plot(history.history["loss"], label="loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel("Epochs", fontdict=fontdict_axis)
        plt.title(model.name + " Run " + str(i) + " - Training Plot - Horizon " + str(h_pars["horizon"]),
                  fontdict=fontdict_title)
        plt.show()

        # Evaluate model
        train = pd.DataFrame(model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
        generate_pit_plot(sc_obs_train_f, get_quantiles(train, np.arange(0.01, 1, 0.01)),
                          model.name + "Training set - Horizon " + str(h_pars["horizon"]), n_bins=50)
        test = pd.DataFrame(model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
        generate_forecast_plots(sc_obs_test_f[::51], test[::51], quantile_levels=np.arange(0.01, 1, 0.01), n=20)
        generate_pit_plot(sc_obs_test_f, get_quantiles(test, np.arange(0.01, 1, 0.01)),
                          model.name + "Test set - Horizon " + str(h_pars["horizon"]), n_bins=50)
        models.append(model)

    # Averaging the models
    for model in models:
        model.trainable = False
    average_model = average_models(models)
    average_model.compile(loss=build_quantile_loss(h_pars["degree"]), optimizer="adam")
    average_model.summary()
    average_model.evaluate(x=sc_ens_test_f, y=sc_obs_test_f)
