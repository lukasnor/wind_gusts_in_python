import json
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 24}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}
figsize = (13, 7)

horizons = [3, 6, 9, 12, 15, 18, 21, 24]
aggregations = ["single", "single+std", "mean+std", "mean", "all"]
variables = ["u100", "v100", "t2m", "sp", "speed", "wind_power"]
variable_selections = [variables]  # or =  list(powerset(variables))[1:]
fixed_params_selections = [{"horizon": a, "variables": b, "aggregation": c} for a, b, c in
                           product(horizons, variable_selections, aggregations)]


def load_hyperparameters_from_folders(path: str) -> pd.DataFrame:
    hps_list = []
    for horizon, aggregation in product(horizons, aggregations):
        with open(path + "horizon:" + str(horizon) + "_agg:" + str(aggregation) + ".json",
                  "r") as file:
            hps_list.append(json.load(file))

    hps = pd.DataFrame(hps_list)
    hps = hps.pivot(index=["horizon", "aggregation"], columns=[])
    return hps


def plot_crps_per_horizon_per_aggregation(evaluation_path: str, plots_path: str = None):
    evaluation = pd.read_csv(evaluation_path).pivot(
        index=["horizon", "aggregation"], columns=[])
    n_runs = len(evaluation.columns) - 1
    for aggregation in aggregations:
        with plt.xkcd():
            plt.figure(figsize=figsize)
            plt.scatter([[horizon for _ in range(n_runs)] for horizon in horizons],
                        evaluation.loc[(slice(None), aggregation), :"run" + str(n_runs)].values,
                        c="blue", label="individual")
            plt.scatter(horizons, evaluation.loc[(slice(None), aggregation), "average"], c="red",
                        label="average")
            plt.legend()
            plt.title("CRPS for aggregation \"" + aggregation + "\"",
                      fontdict=fontdict_title)
            plt.xticks(ticks=horizons, labels=horizons)
            plt.xlabel("Horizons", fontdict=fontdict_axis)
            if plots_path is None:
                plt.show()
            else:
                plt.savefig(plots_path + "crps_per_horizon_per_aggregation/" + aggregation + ".png")


def plot_crps_per_aggregation(evaluation_path: str, plots_path: str = None):
    evaluation = pd.read_csv(evaluation_path)
    evaluation = evaluation.reset_index().pivot(index=["horizon", "aggregation"], columns=[])
    evaluation = evaluation[evaluation.columns.drop("index")]
    evaluation = evaluation.reset_index().pivot(index="aggregation", columns=["horizon"])
    with plt.xkcd():
        plt.figure(figsize=figsize)
        plt.boxplot(evaluation.values.transpose())
        locs, _ = plt.xticks()
        plt.xticks(ticks=locs, labels=evaluation.index)
        plt.xlabel("Aggregations", fontdict=fontdict_axis)
        plt.ylabel("CRPS", fontdict=fontdict_axis)
        plt.title("Performance per Aggregation", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path + "crps_per_aggregation.png")


def plot_crps_per_horizon(evaluation_path: str, plots_path: str = None):
    evaluation = pd.read_csv(evaluation_path)
    evaluation = evaluation.reset_index().pivot(index=["horizon", "aggregation"], columns=[])
    evaluation = evaluation[evaluation.columns.drop("index")]
    evaluation = evaluation.reset_index().pivot(index="horizon", columns=["aggregation"])
    with plt.xkcd():
        plt.figure(figsize=figsize)
        plt.boxplot(evaluation.values.transpose())
        locs, _ = plt.xticks()
        plt.xticks(ticks=locs, labels=evaluation.index)
        plt.xlabel("Horizons", fontdict=fontdict_axis)
        plt.ylabel("CRPS", fontdict=fontdict_axis)
        plt.title("Performance per Horizon", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path + "crps_per_horizon.png")
