import pandas as pd
from keras import layers
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.utils.np_utils import to_categorical
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from src.preprocessing import format_data, import_data, scale_data


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


def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, bin_edges: ndarray,
                            name: str, n=None, path: str = None) -> None:
    pass

if __name__ == "__main__":

    h_pars = {"horizon": 3,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "mean+std",
              "n_bins": 20,  # actually one bin more, since higher than observed values might occur
              "layer_sizes": [20, 15],
              "activations": ["selu", "selu"],

              "batch_size": 25,
              "patience": 25,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed", "wind_speed"]

    # Import the data
    ens_train, ens_test, obs_train, obs_test = import_data(horizon=h_pars["horizon"],
                                                           variables=h_pars["variables"],
                                                           train_split=h_pars["train_split"])

    # Get the bin edges
    bin_edges = binning_scheme(obs_train, h_pars["n_bins"])

    # Make obs categorical
    cat_obs_train = obs_to_categorical(obs_train, bin_edges)
    cat_obs_test = obs_to_categorical(obs_test, bin_edges)

    # Scale the input
    sc_ens_train, \
    sc_ens_test, \
    sc_obs_train, \
    sc_obs_test, \
    scale_dict = scale_data(ens_train,
                            ens_test,
                            obs_train,
                            obs_test,
                            variables=h_pars["train_split"])

    # Format input
    sc_ens_train_f, sc_ens_test_f, _, _ = format_data(sc_ens_train,
                                                      sc_ens_test,
                                                      sc_obs_train,
                                                      sc_obs_test,
                                                      aggregation=h_pars["aggregation"])

    # Build model
    model = get_model("First_model",
                      input_size=len(sc_ens_train_f.columns),
                      layer_sizes=h_pars["layer_sizes"],
                      activations=h_pars["activations"],
                      n_bins=h_pars["n_bins"]+1)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["categorical_accuracy"])

    # Train model
    model.fit(x=sc_ens_train_f,
              y=cat_obs_train,
              batch_size=h_pars["batch_size"],
              epochs=200,
              verbose=1,
              validation_freq=1,
              validation_split=0.1,
              #callbacks=[EarlyStopping(monitor="val_loss",
              #                         patience=h_pars["patience"],
              #                         restore_best_weights=True
              #                         )],
              use_multiprocessing=True
              )

    # Evaluate model
    train = DataFrame(index=sc_ens_train_f.index, data=model.predict(sc_ens_train_f))
    test = DataFrame(index=sc_ens_test_f.index, data=model.predict(sc_ens_test_f))
    print("Evaluation", model.evaluate(x=sc_ens_test_f, y=cat_obs_test))
