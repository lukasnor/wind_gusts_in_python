import pandas as pd
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Model
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from preprocessing import format_data, import_data, scale_data, categorify_data
import tensorflow as tf
from keras.metrics import categorical_crossentropy


# For our data almost equivalent to using quantiles, at least for 20 bins
# From the observations, obtain N+1 bin edges for N bins, excluding the last bin
def binning_scheme(obs: DataFrame, N: int) -> ndarray:
    # Sorted unique observation values
    obs_unique = np.unique(obs.values)

    # Initialize bin edges to contain one observation value
    bin_edges = np.zeros(len(obs_unique) + 1)
    for i in range(1, len(obs_unique)):
        bin_edges[i] = 0.5 * (obs_unique[i - 1] + obs_unique[i])
    bin_edges[-1] = obs_unique[-1]

    # Initialize bins
    bins = [[] for _ in obs_unique]
    for o in obs.values.squeeze():
        for i in range(
                len(bin_edges)):  # this could in log(len(bin_edges)), scine bin_edges are sorted, but meh..
            if o < bin_edges[i]:
                bins[i - 1].append(o)
                break
        else:
            bins[-1].append(o)
    # Count number of observations in each bin
    count = np.array([*map(len, bins)])

    # Reduce bin edges
    while len(bin_edges) > N + 1:
        i_min = count.argmin()  # index of smallest bin
        if i_min == 0:  # left most bin
            bin_edges = np.delete(bin_edges, i_min + 1)
            count[i_min] = count[i_min] + count[i_min + 1]
            count = np.delete(count, i_min + 1)
        elif i_min == len(count) - 1:  # right most bin
            bin_edges = np.delete(bin_edges, i_min)
            count[i_min - 1] = count[i_min - 1] + count[i_min]
            count = np.delete(count, i_min)
        elif count[i_min - 1] < count[i_min + 1]:  # middle bin with left bin smaller
            bin_edges = np.delete(bin_edges, i_min)
            count[i_min - 1] = count[i_min - 1] + count[i_min]
            count = np.delete(count, i_min)
        else:  # middle bin with right bin smaller
            bin_edges = np.delete(bin_edges, i_min + 1)
            count[i_min] = count[i_min] + count[i_min + 1]
            count = np.delete(count, i_min + 1)
    return bin_edges


def get_model(name: str, input_size: int, layer_sizes: [int], activations: [str],
              n_bins: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=input_size, name="ens_input")

    # Hidden layers
    x = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden0")(ens_input)
    for i in range(len(layer_sizes) - 1):
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1],
                         name="hidden" + str(i + 1))(x)

    # Output
    output = layers.Dense(name="output", units=n_bins, activation="softmax")(x)  # we want a density

    # Model
    return Model(name=name, inputs=ens_input, outputs=output)


def build_hen_crps(bin_edges: np.ndarray):
    N = len(bin_edges) - 1
    b = tf.constant(bin_edges, dtype="float", shape=(1, len(bin_edges)))
    b_minus = b[:, :-1]
    b_plus = b[:, 1:]
    d = b_plus - b_minus

    def crps(y_true, y_pred):
        y_tilde = tf.minimum(b_plus, tf.maximum(b_minus, y_true))
        L = tf.cumsum(y_pred, axis=1, exclusive=True)
        first = tf.pow(L, 2) * (y_tilde - b_minus)
        second = tf.pow(1 - L, 2) * (b_plus - y_tilde)
        third = y_pred * tf.pow(y_tilde - b_minus, 2) / d
        forth = y_pred * d * (L - 1 + y_pred / 3)
        fifth = tf.abs(y_true - tf.minimum(b[0, N], tf.maximum(b[0, 0], y_true)))
        return tf.reduce_mean(
            tf.expand_dims(tf.reduce_sum(first + second + third + forth, axis=1), axis=1) + fifth)

    return crps


# Build the hen crps for the observation data, in which the first column is the exact observation
def build_wrapped_hen_crps(bin_edges: np.ndarray):
    crps_prime = build_hen_crps(bin_edges)

    def crps(y_true, y_pred):
        return crps_prime(tf.expand_dims(y_true[:, 0], axis=1), y_pred)

    return crps


# build the crossentropy loss for the observation data, in which the second to last columns are the
# categorical vectors of the observation
def build_wrapped_crossentropy_loss():
    def loss(y_true, y_pred):
        return categorical_crossentropy(y_true[:, 1:], y_pred)

    return loss


# Quantile function of a piecewise linear distribution function
# alpha: ndarray with shape (k,) ; quantile levels between 0.0 and 1.0
# bin_edges: ndarray with shape (m,) ; bin edges as floats
# bin_probs: ndarray with shape (n,) ; bin probabilities between 0.0 and 1.0, summing to 1.0
# returns ndarray with shape (k,) ; quantiles to the levels in alpha, numerically well-behaved
def quantile_function(alpha: ndarray, bin_edges: ndarray, bin_probs: ndarray):
    beta = np.expand_dims(alpha, 1)  # (k, 1)
    f = np.expand_dims(bin_probs.cumsum(), 0)  # (1, m)
    g = np.insert(bin_probs.cumsum(), 0, 0.0)[:-1]
    booleans = beta>=f  # (k, m)
    i = np.argmin(booleans, axis=1)  # (k,)
    i[booleans[:, -1] == True] = len(bin_probs)-1  # needed for an all true row in booleans
    return np.minimum(bin_edges[i] + (bin_edges[i + 1] - bin_edges[i]) * (
                alpha - g[i]) / bin_probs[i], bin_edges[-1])


def test_quantile_function():
    bin_edges = np.array([1., 2., 3., 4.])
    bin_probs = np.array([0.25, 0.5, 0.249])
    alpha = np.array([0.0, 0.2, 0.25, 0.5, 0.75, 0.998, 0.999, 1.0])
    qs = quantile_function(alpha, bin_edges, bin_probs)


# Not ready yet
def vincentivize_forecasts(bin_probs_list: [DataFrame]) -> DataFrame:
    levels = pd.concat(map(lambda ps: ps.cumsum(axis=1).round(3), bin_probs_list), axis=1)
    levels_sorted = DataFrame(np.sort(levels, axis=1), index=levels.index)
    levels_sorted = DataFrame(
        [levels_sorted.iloc[i, :].unique() for i in range(len(levels_sorted))],
        index=levels_sorted.index)


def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, bin_edges: ndarray,
                            name: str, n=None, path: str = None) -> None:
    pass


if __name__=="__main__":

    h_pars = {"horizon": 3,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "mean",
              "n_bins": 20,
              "layer_sizes": [30, 20, 20],
              "activations": ["selu", "selu", "selu"],

              "batch_size": 25,
              "patience": 25,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed", "wind_power"]

    # Import the data
    ens_train, \
    ens_test, \
    obs_train, \
    obs_test = import_data(horizon=h_pars["horizon"],
                           variables=h_pars["variables"],
                           train_split=h_pars["train_split"])

    # Get the bin edges
    bin_edges = binning_scheme(obs_train, h_pars["n_bins"])

    # # Make observations categorical
    # # drop last category, since no train value is outside last bin
    # cat_obs_train = obs_to_categorical(obs_train, bin_edges).iloc[:, :-1]
    # # Merge outliers into the last bin in the test data
    # cat_obs_test = obs_to_categorical(obs_test, bin_edges)
    # cat_obs_test.iloc[:, -2] = cat_obs_test.iloc[:, -2:].sum(axis=1)
    # cat_obs_test = cat_obs_test.iloc[:, :-1]
    #
    # # Merge numerical and categorical observations
    # merged_obs_train = pd.concat([obs_train, cat_obs_train], axis=1)
    # merged_obs_test = pd.concat([obs_test, cat_obs_test], axis=1)

    # Scale the input
    sc_ens_train, \
    sc_ens_test, \
    sc_obs_train, \
    sc_obs_test, \
    input_scalers, \
    output_scalers = scale_data(ens_train,
                                ens_test,
                                obs_train,
                                obs_test,
                                # do not scale "wind_power"
                                input_variables=["u100", "v100", "t2m", "sp", "speed"],
                                output_variables=[])  # do not scale output variables

    # Format input
    sc_ens_train_f, \
    sc_ens_test_f, \
    sc_obs_train_f, \
    sc_obs_test_f = format_data(sc_ens_train,
                                sc_ens_test,
                                sc_obs_train,
                                sc_obs_test,
                                aggregation=h_pars["aggregation"])

    # Categorify wind_power data
    sc_ens_train_fc, \
    sc_ens_test_fc, \
    sc_obs_train_fc, \
    sc_obs_test_fc = categorify_data(sc_ens_train_f,
                                     sc_ens_test_f,
                                     sc_obs_train_f,
                                     sc_obs_test_f,
                                     bin_edges)
    # Drop the categorical data from the inputs
    sc_ens_train_fc = sc_ens_train_fc.iloc[:, : (-1) * h_pars["n_bins"]]
    sc_ens_test_fc = sc_ens_test_fc.iloc[:, : (-1) * h_pars["n_bins"]]

    models = []
    bin_probs_list = []
    for _ in range(2):
        # Build model
        model = get_model("First_model",
                          input_size=len(sc_ens_train_fc.columns),
                          layer_sizes=h_pars["layer_sizes"],
                          activations=h_pars["activations"],
                          n_bins=h_pars["n_bins"])
        model.compile(optimizer="adam",
                      loss=build_wrapped_crossentropy_loss(),
                      metrics=[build_wrapped_hen_crps(bin_edges)])

        # Train model
        model.fit(x=sc_ens_train_fc,
                  y=sc_obs_train_fc,
                  batch_size=h_pars["batch_size"],
                  epochs=20,
                  verbose=1,
                  validation_freq=1,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(monitor="val_loss",
                                           patience=h_pars["patience"],
                                           restore_best_weights=True
                                           )],
                  use_multiprocessing=True
                  )

        # Evaluate model
        crps = build_hen_crps(bin_edges)
        train = DataFrame(index=sc_ens_train_fc.index, data=model.predict(sc_ens_train_fc))
        test = DataFrame(index=sc_ens_test_fc.index, data=model.predict(sc_ens_test_fc))
        print("Evaluation - CRPS:", crps(obs_test.values, test.values))

        models.append(model)
        bin_probs_list.append(test)
