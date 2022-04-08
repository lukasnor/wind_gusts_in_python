import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 18}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}

horizons = [3, 6, 9, 12, 15, 18, 21, 24]
aggregations = ["single", "single+std", "mean", "all"]


def generate_simple_plot(df, x, y):
    plt.scatter(df[x], df[y])
    plt.xlim(0, df[x].max() + 1)
    plt.ylim(0, df[y].max() + 1)
    plt.xlabel(x, fontdict=fontdict_axis)
    plt.ylabel(y, fontdict=fontdict_axis)
    plt.title(x + " vs " + y, fontdict=fontdict_title)
    plt.show()


def load_hyperparameters_from_folders():
    hps_list = []
    for horizon, aggregation in product(horizons, aggregations):
        with open("../results/tuning/horizon:" + str(horizon) + "_agg:" + str(aggregation) + "/best_hps.json",
                  "r") as file:
            hps_list.append(json.load(file))

    hps = pd.DataFrame(hps_list)
    hps["run"] = 1
    hps.pivot(index=["horizon", "aggregation", "run"],
              columns=["input_size", "batch_size", "learning_rate", "activation", "degree", "depth", "layer0_size",
                       "layer1_size", "layer2_size", "layer3_size", "layer4_size"])
    return hps


def analysis_by_plots():
    hyperparameters = load_hyperparameters_from_folders()
    hyperparameters.info()
    generate_simple_plot(hyperparameters, "horizon", "degree")
    generate_simple_plot(hyperparameters, "degree", "depth")
    # Complexity of neural net, i.e. number of neurons
