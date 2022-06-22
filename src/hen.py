import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from src.preprocessing import preprocess_data, format_data, import_data, scale_data


# From the observations, obtain N+1 bin edges for N bins
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
        for i in range(len(bin_edges)):  # this could in log(len(bin_edges)), scine bin_edges are sorted, but meh..
            if o < bin_edges[i]:
                bins[i - 1].append(o)
                break
        else:
            bins[-1].append(o)
    # Count number of observations in each bin
    count = np.array([*map(lambda bin: len(bin), bins)])

    # Reduce bin edges
    while len(bin_edges) > N + 1:
        i_min = count.argmin()
        if i_min == 0:  # left most bin
            bin_edges = np.delete(bin_edges, i_min+1)
            count[i_min] = count[i_min] + count[i_min+1]
            count = np.delete(count, i_min+1)
        elif i_min == len(count)-1:  # right most bin
            bin_edges = np.delete(bin_edges, i_min)
            count[i_min-1] = count[i_min-1] + count[i_min]
            count = np.delete(count, i_min)
        elif count[i_min-1] < count[i_min+1]:  # middle bin with left bin smaller
            bin_edges = np.delete(bin_edges, i_min)
            count[i_min-1] = count[i_min-1]+count[i_min]
            count = np.delete(count, i_min)
        else:  # middle bin with right bin smaller
            bin_edges = np.delete(bin_edges, i_min+1)
            count[i_min] = count[i_min] + count[i_min+1]
            count = np.delete(count, i_min+1)
    return bin_edges


if __name__ == "__main__":

    h_pars = {"horizon": 3,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "mean+std",
              "degree": 12,
              "layer_sizes": [20, 15],
              "activations": ["selu", "selu", "selu"],

              "batch_size": 25,
              "patience": 50,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed"]

    # Import the data and scale it..
    ens_train, ens_test, obs_train, obs_test = import_data(h_pars)

    # Test bin edges algorithm
    bin_egdes = binning_scheme(obs_train, 20)
    print(bin_egdes)