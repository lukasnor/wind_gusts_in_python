import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Model, layers
from keras.callbacks import EarlyStopping
from scipy.special import binom
from preprocessing import preprocess_data, format_data

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 24}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}
figsize = (13, 7)


# Construction of the local model
def get_local_model(name: str, input_size: int, n_loc: int, n_emb: int, layer_sizes: [int],
                    activations: [str],
                    degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=input_size, name="ens_input")
    emb_input = layers.Input(shape=1, name="emb_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc,
                                              output_dim=n_emb,
                                              input_shape=1)(emb_input)
    station_embedding_flat = layers.Flatten()(station_embedding_part)
    # Merge Inputs
    merged = layers.Concatenate(name="merged")([ens_input, station_embedding_flat])

    # Hidden layers
    hidden1 = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden1")(merged)
    hidden2 = layers.Dense(units=layer_sizes[1], activation=activations[1], name="hidden2")(hidden1)

    # Output
    output = layers.Dense(units=degree + 1, activation="softplus")(
        hidden2)  # smooth, non-negative and proportional

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
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1],
                         name="hidden" + str(i + 1))(x)

    # Output
    output = layers.Dense(name="output", units=degree + 1, activation="softplus")(
        x)  # smooth, non-negative and proportional

    # Model
    return Model(name=name, inputs=ens_input, outputs=output)


# For a list of models of same type, construct the average model
def average_models(models: [Model], name) -> Model:
    common_input = layers.Input(name="common_input", shape=models[0].input.shape[1])
    average_output = layers.Average(name="average")([model(common_input) for model in models])
    return Model(name=name, inputs=common_input, outputs=average_output)


# Construction of the loss function
def build_quantile_loss(degree: int):  # -> Loss function
    lql = tf.constant(np.linspace(0.01, 0.99, 99),
                      dtype="float32")  # 1% to 99% quantile levels for the loss, equidistant
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(),
                    dtype="float32")  # Bernstein polynomials to interpolate the CDF

    # Multi-quantile loss: sum over all quantile losses for levels in lql
    def quantile_loss(y_true, y_pred):
        quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = y_true - quantiles  # no expand dims
        err1 = error * tf.expand_dims(lql - 1, 0)
        err2 = error * tf.expand_dims(lql, 0)
        loss = tf.maximum(err1, err2)
        return tf.reduce_sum(loss, axis=1)

    return quantile_loss


# WARNING: DO NOT USE FOR TRAINING!
# Not numerically stable under constant forecasts, for evaluation only
# CRPS loss implementation for Bernstein Quantile Networks
# Scale parameter for ease of comparison to work of others
def build_crps_loss(degree: int, scale=1.0):
    lql = tf.constant(np.arange(0.0, 1.01, 0.01),
                      dtype="float32")  # 1% to 99% quantile levels for the loss, equidistant
    # Bernstein polynomials to interpolate the CDF
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")
    # Derivative of the Bernstein polynomials, B'_0 and B'_degree must be considered separately
    B_prime = tf.transpose(tf.constant(np.array(
        [-degree * np.power(1 - lql, degree - 1)]  # B'_0
        + [binom(degree, j) * np.power(lql, j - 1) * np.power(1 - lql, degree - j - 1)
           * (j - degree * lql)  # B'_j
           for j in range(1, degree)]
        + [degree * np.power(lql, degree - 1)]  # B'_degree
    ), dtype="float32"))

    def crps(y_true, y_pred):
        q = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        q_prime = tf.transpose(tf.tensordot(B_prime, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = q - y_true  # no expand dims
        square = tf.square(lql - tf.experimental.numpy.heaviside(error, 1))
        integrand = tf.multiply(square, q_prime)
        return y_true * scale * tf.reduce_mean(integrand, axis=1)

    return crps


def build_crps_loss2(degree: int, min=0.0, max=1.0, eps=0.0):
    lql = tf.constant(np.arange(0.0, 1.01, 0.01),
                      dtype="float32")  # 1% to 99% quantile levels for the loss, equidistant
    # Bernstein polynomials to interpolate the CDF
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")
    B_prime = tf.transpose(tf.constant(np.array(
        [-degree * np.power(1 - lql, degree - 1)]  # B'_0
        + [binom(degree, j) * np.power(lql, j - 1) * np.power(1 - lql, degree - j - 1)
           * (j - degree * lql)  # B'_j
           for j in range(1, degree)]
        + [degree * np.power(lql, degree - 1)]  # B'_degree
    ), dtype="float32"))

    def crps(y_true, y_pred):
        q = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        q_prime = tf.transpose(tf.tensordot(B_prime, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = q - y_true  # no expand dims
        square = tf.square(error)
        integrand = tf.multiply(square, tf.pow(tf.abs(q_prime) + eps, -0.75))
        return (max - min) * tf.reduce_mean(integrand, axis=1)

    return crps


def build_crps_loss3(degree: int, min=0.0, max=1.0):
    # random quantile levels
    lql = tf.constant(np.linspace(0, 1, 10001))
    # Bernstein polynomials to interpolate the CDF
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")
    lql2 = tf.random.shuffle(lql)
    B2 = tf.constant(np.array(
        [binom(degree, j) * np.power(lql2, j) * np.power(1 - lql2, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")

    def crps(y_true, y_pred):
        # X
        q = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        # X*
        q2 = tf.transpose(tf.tensordot(B2, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        # |X-x|
        error = (max - min) * tf.abs(q - y_true)
        # |X-X*|
        error2 = (max - min) * tf.abs(q - q2)
        return tf.reduce_mean(error, axis=1) - 0.5 * tf.reduce_mean(error2, axis=1)

    return crps


# Custom loss as the sum of the quantile loss and the crps
def build_custom_loss(degree: int):
    lql = tf.constant(np.arange(0.0, 1.01, 0.01),
                      dtype="float32")  # 1% to 99% quantile levels for the loss, equidistant
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(),
                    dtype="float32")  # Bernstein polynomials to interpolate the CDF
    B_prime = tf.transpose(tf.constant(np.array([-degree * np.power(1 - lql, degree - 1)] + [
        binom(degree, j) * np.power(lql, j - 1) * np.power(1 - lql, degree - j - 1) * (
                j - degree * lql) for j in
        range(1, degree)] + [degree * np.power(lql, degree - 1)]), dtype="float32"))

    def custom_loss(y_true, y_pred):
        quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = y_true - quantiles  # no expand dims
        err1 = error * tf.expand_dims(lql - 1, 0)
        err2 = error * tf.expand_dims(lql, 0)
        loss = tf.maximum(err1, err2)
        q_prime = tf.transpose(tf.tensordot(B_prime, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        error = quantiles - y_true  # no expand dims
        square = tf.square(lql - tf.experimental.numpy.heaviside(error, 1))
        integrand = tf.multiply(square, q_prime)
        return tf.reduce_mean(integrand, axis=1) + tf.reduce_sum(loss, axis=1)

    return custom_loss


# From the model prediction, i.e. the increments of the coefficients of bernstein pols, calculate the actual quantiles
def get_quantiles(y_pred: pd.DataFrame, quantile_levels: np.ndarray) -> pd.DataFrame:
    degree = len(y_pred.columns) - 1
    bernsteins = np.array(
        [binom(degree, j) * np.power(quantile_levels, j) * np.power(1 - quantile_levels, degree - j)
         for j in
         range(degree + 1)]).transpose()
    return pd.DataFrame(
        np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]])),
        index=y_pred.index)


# Calculate the rank of the observation in the quantile forecast
def get_rank(obs: pd.DataFrame, quantiles: pd.DataFrame) -> pd.Series:
    return pd.concat([obs, quantiles], axis=1).rank(axis=1).iloc[:, 0].rename("rank").astype("int")


# Path with a trailing "/"!
def generate_histogram_plot(obs: pd.DataFrame, f: pd.DataFrame, name: str, bins: int = 10,
                            path: str = None, filename: str = "rankhistogram") -> None:
    if bins < 2:
        raise Exception(
            "More than one bin is necessary for a sensable rank histogram. 'bins' = " + str(bins))
    quantile_levels = np.linspace(1 / bins, 1, bins - 1, False)  # Quantile levels / inner bin walls
    quantiles = get_quantiles(f, quantile_levels)
    ranks = get_rank(obs, quantiles)
    plt.figure(figsize=figsize)
    plt.hist(ranks, bins=np.arange(0.5, bins + 1), weights=np.ones_like(ranks) / ranks.size)
    plt.hlines(1 / bins, linestyles="dashed", color="grey", xmin=0.5, xmax=bins + 0.5)
    # TODO: y-achse bis festen Wert 0.5 z.B. für Vergleichbarkeit
    plt.xlabel("Ranks", fontdict=fontdict_axis)
    plt.xticks(np.arange(1, bins + 1), np.arange(1, bins + 1))
    plt.ylim(bottom=0, top=0.25)
    plt.title(name, fontdict=fontdict_title)
    if path is None:
        plt.show()
    else:
        plt.savefig(path + filename + "_" + str(i) + ".png")


# Path with a trailing "/"!
def generate_pit_plot(obs: pd.DataFrame, quantiles: pd.DataFrame, name: str, n_bins: int = 10,
                      path: str = None, filename: str = "rankhistogram") -> None:
    ranks = get_rank(obs, quantiles)
    plt.figure(figsize=figsize)
    plt.hist(ranks, bins=n_bins, weights=np.ones_like(ranks) / ranks.size)
    plt.hlines(1 / n_bins, linestyles="dashed", color="grey", xmin=0,
               xmax=len(quantiles.columns) + 1)
    plt.xlabel("Ranks", fontdict=fontdict_axis)
    plt.title("Rank Histogram - " + name, fontdict=fontdict_title)
    if path is None:
        plt.show()
    else:
        plt.savefig(path + filename + "_" + str(i) + ".png")


# Path with a trailing "/"!
def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, quantile_levels: np.ndarray,
                            name: str, n=None, path: str = None,
                            filename: str = "forecast") -> None:
    q = get_quantiles(y_pred, quantile_levels=quantile_levels)
    if n is None:
        n = y_true.shape[0]
    for i in range(n):
        plt.figure(figsize=figsize)
        plt.plot(q.iloc[(i, slice(None))], quantile_levels, color="blue", label="forecast")
        plt.vlines(y_true.iloc[i], ymin=quantile_levels.min(), ymax=quantile_levels.max(),
                   label="observation", color="red", linestyles="dashed")
        plt.xlim(left=0.0, right=max(1.0, plt.axis()[1]))
        plt.title(name + " - Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(path + filename + "_" + str(i) + ".png")


if __name__ == "__main__":
    # Get the data in a processed form
    h_pars = {"horizon": 3,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "mean+std",
              "degree": 10,
              "layer_sizes": [20, 15, 10],
              "activations": ["selu", "selu", "selu"],

              "batch_size": 25,
              "patience": 27,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed", "wind_power"]

    # Import the data
    sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers \
        = preprocess_data(horizon=h_pars["horizon"],
                          train_variables=h_pars["variables"],
                          train_split=h_pars["train_split"],
                          input_variables=h_pars["variables"],
                          output_variables=None)
    obs_scaler = output_scalers["wind_power"]
    obs_max = obs_scaler.data_max_
    obs_min = obs_scaler.data_min_
    #obs_max = 1.0
    #obs_min = 0.0
    # Format the data
    sc_ens_train_f, \
    sc_ens_test_f, \
    sc_obs_train_f, \
    sc_obs_test_f = format_data(sc_ens_train,
                                sc_ens_test,
                                sc_obs_train,
                                sc_obs_test,
                                h_pars["aggregation"])

    # Average over models
    models = []
    for i in range(10):
        # Build model
        model = get_model(name="Foo" + str(i),
                          input_size=len(sc_ens_train_f.columns),
                          layer_sizes=h_pars["layer_sizes"],
                          activations=h_pars["activations"],
                          degree=h_pars["degree"])
        model.compile(optimizer="adam",
                      loss=build_quantile_loss(h_pars["degree"]),
                      metrics=[build_crps_loss3(h_pars["degree"], obs_min, obs_max)]
                      )
        # Fit model
        history = model.fit(y=sc_obs_train_f,
                            x=sc_ens_train_f,
                            batch_size=h_pars["batch_size"],
                            epochs=300,
                            verbose=1,
                            validation_freq=1,
                            validation_split=0.1,
                            callbacks=[EarlyStopping(monitor="val_crps",
                                                     patience=h_pars["patience"],
                                                     restore_best_weights=True
                                                     )],
                            use_multiprocessing=True
                            )
        # Plot the learning curves
        with plt.xkcd():
            plt.figure(figsize=figsize)
            plt.plot(history.history["loss"][5:], label="loss")
            plt.plot(history.history["val_loss"][5:], label="val_loss")
            plt.plot(history.history["val_crps"][5:], label="val_crps")
            plt.legend()
            plt.xlabel("Epochs", fontdict=fontdict_axis)
            plt.title(
                model.name + " Run " + str(i) + " - Training Plot - Horizon " + str(
                    h_pars["horizon"]),
                fontdict=fontdict_title)
            plt.show()

        # Evaluate model
        train = pd.DataFrame(model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
        # Newer and better histogram plot
        with plt.xkcd():
            generate_histogram_plot(sc_obs_train_f, train, "Rank Histogram of Training data", 21)
        test = pd.DataFrame(model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
        # Forecast plots
        with plt.xkcd():
            generate_forecast_plots(sc_obs_test_f[::51], test[::51], name="Forecast Plot",
                                    quantile_levels=np.linspace(0, 1, 101), n=1)
        # Test data forecast plots
        with plt.xkcd():
            generate_histogram_plot(sc_obs_test_f, test, "Rank Histogram of Test data", 21)
        models.append(model)

    if len(models) == 1:
        models += [models[0]]

    # Averaging the models
    for model in models:
        model.trainable = False
    average_model = average_models(models, name="average_model")
    average_model.compile(loss=build_crps_loss3(h_pars["degree"], obs_min, obs_max),
                          optimizer="adam")
    # Evaluate the averaged model
    average_model.evaluate(x=sc_ens_test_f, y=sc_obs_test_f)
    test = pd.DataFrame(average_model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
    with plt.xkcd():
        generate_histogram_plot(sc_obs_test_f, test,
                                "Rank Histogram - Test data\n" + "Horizon " + str(
                                    h_pars["horizon"]) + " - Aggregation " + h_pars["aggregation"],
                                21)
