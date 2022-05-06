import json
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.optimizer_v2 import adam

from bqn import preprocess_data, \
    format_data, get_model, build_quantile_loss, build_crps_loss3, average_models, get_quantiles, \
    generate_pit_plot, generate_forecast_plots

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 24}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}

horizons = [3, 6, 9, 12, 15, 18, 21, 24]
aggregations = ["single", "single+std", "mean+std", "all"]
variables = ["u100", "v100", "t2m", "sp", "speed"]
variable_selections = [variables]  # or =  list(powerset(variables))[1:]
fixed_params_selections = [{"horizon": a, "variables": b, "aggregation": c} for a, b, c in
                           product(horizons, variable_selections, aggregations)]


def generate_simple_plot(df, x, y):
    plt.scatter(df[x], df[y])
    plt.xlim(0, df[x].max() + 1)
    plt.ylim(0, df[y].max() + 1)
    plt.xlabel(x, fontdict=fontdict_axis)
    plt.ylabel(y, fontdict=fontdict_axis)
    plt.title(x + " vs " + y, fontdict=fontdict_title)
    plt.show()


def load_hyperparameters_from_folders(path: str) -> pd.DataFrame:
    hps_list = []
    for horizon, aggregation in product(horizons, aggregations):
        with open(path + "horizon:" + str(horizon) + "_agg:" + str(aggregation) + ".json",
                  "r") as file:
            hps_list.append(json.load(file))

    hps = pd.DataFrame(hps_list)
    hps = hps.pivot(index=["horizon", "aggregation"],
                    # columns=["input_size", "batch_size", "learning_rate", "activation", "degree", "depth",
                    #          "layer0_size",
                    #          "layer1_size", "layer2_size", "layer3_size", "layer4_size"])
                    columns=[])
    return hps


def analysis_by_plots():
    hyperparameters = load_hyperparameters_from_folders(path="../results/bqn/hps/")
    hyperparameters.info()
    generate_simple_plot(hyperparameters, "horizon", "degree")
    generate_simple_plot(hyperparameters, "degree", "depth")
    # Complexity of neural net, i.e. number of neurons


def evaluate_best_hps():
    # Average 5 models
    runs = 5

    # Import the best hps
    hps = load_hyperparameters_from_folders("../results/bqn/hps/")
    evaluation = pd.DataFrame(index=hps.index,
                              columns=["run" + str(i + 1) for i in range(runs)] + ["average"])

    fixed_params = {"variables": variables, "train_split": 0.85, "patience": 27}
    # horizons =[3]
    # aggregations = ["single"]
    for horizon in horizons:
        fixed_params["horizon"] = horizon
        # Import data
        sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, scale_dict \
            = preprocess_data(fixed_params)
        obs_scaler = scale_dict["wind_power"]
        obs_max = obs_scaler.data_max_[0]
        obs_min = obs_scaler.data_min_[0]

        for aggregation in aggregations:
            index = (horizon, aggregation)
            fixed_params["aggregation"] = aggregation

            if aggregation in ["single", "single+std"]:
                batch_size = 100
            else:
                batch_size = 5
            fixed_params["batch_size"] = batch_size
            fixed_params["patience"] = 27

            layer_sizes = [hps.loc[index, "layer" + str(j) + "_size"] for j in
                           range(hps.loc[index, "depth"])]
            layer_sizes.sort(reverse=True)
            fixed_params["layer_sizes"] = layer_sizes
            fixed_params["activations"] = [hps.loc[index, "activation"]
                                           for _ in range(len(layer_sizes))]
            fixed_params["degree"] = hps.loc[index, "degree"]
            fixed_params["learning_rate"] = hps.loc[index, "learning_rate"]

            # Format data according to aggregation method
            (sc_ens_train_f,
             sc_ens_test_f,
             sc_obs_train_f,
             sc_obs_test_f) = format_data(fixed_params,
                                          sc_ens_train,
                                          sc_ens_test,
                                          sc_obs_train,
                                          sc_obs_test)
            models = []
            for i in range(runs):
                model = get_model(name="run" + str(i + 1),
                                  input_size=len(sc_ens_train_f.columns),
                                  layer_sizes=fixed_params["layer_sizes"],
                                  activations=fixed_params["activations"],
                                  degree=fixed_params["degree"]
                                  )
                model.compile(optimizer=adam.Adam(fixed_params["learning_rate"]),
                              loss=build_quantile_loss(fixed_params["degree"]),
                              metrics=[build_crps_loss3(fixed_params["degree"],
                                                        obs_min, obs_max)]
                              )
                history = model.fit(x=sc_ens_train_f,
                                    y=sc_obs_train_f,
                                    batch_size=fixed_params["batch_size"],
                                    epochs=150,
                                    verbose=1,
                                    validation_freq=1,
                                    validation_split=0.1,
                                    callbacks=[EarlyStopping(monitor="val_crps",
                                                             patience=fixed_params["patience"],
                                                             restore_best_weights=True)],
                                    use_multiprocessing=True
                                    )
                model.trainable = False

                # Evaluation
                model.compile(optimizer="adam",
                              loss=build_crps_loss3(fixed_params["degree"], obs_min, obs_max))
                evaluation.loc[index, "run" + str(i + 1)] \
                    = model.evaluate(x=sc_ens_test_f, y=sc_obs_test_f)
                print(evaluation)
                models.append(model)

            average_model = average_models(models, name=str(horizon)+aggregation)
            average_model.compile(loss=build_crps_loss3(fixed_params["degree"],
                                                        obs_min, obs_max)
                                  )

            # Generate plots for average_model
            train = pd.DataFrame(average_model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
            test = pd.DataFrame(average_model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
            quantile_levels = np.arange(0.0, 1.01, 0.01)
            quantiles_train = get_quantiles(train, quantile_levels)
            quantiles_test = get_quantiles(test, quantile_levels)
            with plt.xkcd():
                generate_pit_plot(obs=sc_obs_train_f,
                                  quantiles=quantiles_train,
                                  name=str(horizon) + " - " + aggregation + " - Train",
                                  n_bins=50,
                                  path="../results/bqn/plots/rankhistograms/horizon:" + str(
                                      horizon) + "_agg:" + aggregation + "_train.png")
                generate_pit_plot(obs=sc_obs_test_f,
                                  quantiles=quantiles_test,
                                  name=str(horizon) + " - " + aggregation + " - Test",
                                  n_bins=50,
                                  path="../results/bqn/plots/rankhistograms/horizon:" + str(
                                      horizon) + "_agg:" + aggregation + "_test.png")
                generate_forecast_plots(y_true=sc_obs_test_f,
                                        y_pred=test,
                                        quantile_levels=quantile_levels,
                                        name="Test - Horizon " + str(
                                            horizon) + " - Aggregation " + aggregation,
                                        n=1,
                                        path="../results/bqn/plots/forecasts/horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_test.png")
            evaluation.loc[index, "average"] = average_model.evaluate(x=sc_ens_test_f,
                                                                      y=sc_obs_test_f)
            print(evaluation)
            # Make model persistent for future evaluation
            average_model.save(
                "../results/bqn/models/horizon:" + str(horizon) + "_agg:" + aggregation + ".h5")

    evaluation.to_csv("../results/bqn/crps_evaluation.csv")


def plot_crps_per_horizon_per_aggregation(plots_path=None):
    evaluation = pd.read_csv("../results/bqn/crps_evaluation.csv")
    evaluation = evaluation.reset_index().pivot(index=["horizon", "aggregation"], columns=[])
    evaluation = evaluation[evaluation.columns.drop("index")]
    for aggregation in aggregations:
        with plt.xkcd():
            plt.figure(figsize=(13, 7))
            plt.scatter([[horizon for _ in range(5)] for horizon in horizons],
                        evaluation.loc[(slice(None), aggregation), :"run5"].values, c="blue",
                        label="individual")
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


def plot_crps_per_aggregation(plots_path=None):
    evaluation = pd.read_csv("../results/bqn/crps_evaluation.csv")
    evaluation = evaluation.reset_index().pivot(index=["horizon", "aggregation"], columns=[])
    evaluation = evaluation[evaluation.columns.drop("index")]
    evaluation = evaluation.reset_index().pivot(index="aggregation", columns=["horizon"])
    with plt.xkcd():
        plt.figure(figsize=(13, 7))
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


def plot_crps_per_horizon(plots_path=None):
    evaluation = pd.read_csv("../results/bqn/crps_evaluation.csv")
    evaluation = evaluation.reset_index().pivot(index=["horizon", "aggregation"], columns=[])
    evaluation = evaluation[evaluation.columns.drop("index")]
    evaluation = evaluation.reset_index().pivot(index="horizon", columns=["aggregation"])
    with plt.xkcd():
        plt.figure(figsize=(13, 7))
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


if __name__ == "__main__":
    # plots_path = "../results/bqn/plots/"
    # plot_crps_per_horizon_per_aggregation(plots_path)
    # plot_crps_per_horizon(plots_path)
    # plot_crps_per_aggregation(plots_path)
    evaluate_best_hps()
