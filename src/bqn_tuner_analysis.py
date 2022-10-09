import json
from itertools import product

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.optimizer_v2 import adam

from bqn import get_model, build_quantile_loss, build_crps_loss3, average_models, \
    generate_forecast_plots, generate_histogram_plot
from preprocessing import preprocess_data, format_data
from tuner_helper import plot_crps_per_horizon_per_aggregation, plot_crps_per_aggregation, \
    plot_crps_per_horizon, load_hyperparameters_from_folders

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


def generate_simple_plot(df, x, y):
    plt.scatter(df[x], df[y])
    plt.xlim(0, df[x].max() + 1)
    plt.ylim(0, df[y].max() + 1)
    plt.xlabel(x, fontdict=fontdict_axis)
    plt.ylabel(y, fontdict=fontdict_axis)
    plt.title(x + " vs " + y, fontdict=fontdict_title)
    plt.show()


def analysis_by_plots():
    hyperparameters = load_hyperparameters_from_folders(path="../results/bqn/hps/")
    hyperparameters.info()
    generate_simple_plot(hyperparameters, "horizon", "degree")
    generate_simple_plot(hyperparameters, "degree", "depth")
    # Complexity of neural net, i.e. number of neurons


def evaluate_best_hps():
    # Average 10 models
    n_runs = 10

    # Import the best hps
    hps = load_hyperparameters_from_folders("../results/bqn/hps/")
    evaluation = pd.DataFrame(index=hps.index,
                              columns=["run" + str(i + 1) for i in range(n_runs)] + ["average"])

    fixed_params = {"variables": variables, "train_split": 0.85, "data_set": "sweden", "patience": 27}
    # horizons = [12]
    # aggregations = ["single"]
    # evaluation = pd.read_csv("../results/bqn/crps_evaluation.csv").pivot(
    #     index=["horizon", "aggregation"], columns=[])
    for horizon in horizons:
        fixed_params["horizon"] = horizon
        # Import data
        sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers \
            = preprocess_data(horizon, variables, train_split=fixed_params["train_split"],
                              data_set=fixed_params["data_set"])
        obs_scaler = output_scalers["wind_power"]
        obs_max = obs_scaler.data_max_[0]
        obs_min = obs_scaler.data_min_[0]

        for aggregation in aggregations:
            index = (horizon, aggregation)
            fixed_params["aggregation"] = aggregation
            fixed_params["batch_size"] = hps.loc[index, "batch_size"]
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
             sc_obs_test_f) = format_data(sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test,
                                          aggregation, fixed_params["data_set"])
            models = []
            for i in range(n_runs):
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

            average_model = average_models(models, name="average_model")
            average_model.compile(loss=build_crps_loss3(fixed_params["degree"],
                                                        obs_min, obs_max)
                                  )

            # Generate plots for average_model
            train = pd.DataFrame(average_model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
            test = pd.DataFrame(average_model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
            quantile_levels = np.arange(0.0, 1.01, 0.01)
            with plt.xkcd():
                generate_histogram_plot(obs=sc_obs_train_f,
                                        f=train,
                                        name="Rank Histogram - " + str(
                                            horizon) + " - " + aggregation + " - Train",
                                        bins=21,
                                        path="../results/bqn/plots/rankhistograms/",
                                        filename="horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_train"
                                        )
                generate_histogram_plot(obs=sc_obs_test_f,
                                        f=test,
                                        name="Rank Histogram - " + str(
                                            horizon) + " - " + aggregation + " - Test",
                                        bins=21,
                                        path="../results/bqn/plots/rankhistograms/",
                                        filename="horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_test"
                                        )
                generate_forecast_plots(y_true=sc_obs_test_f[::51],
                                        y_pred=test[::51],
                                        quantile_levels=quantile_levels,
                                        name="Test - Horizon " + str(
                                            horizon) + " - Aggregation " + aggregation,
                                        n=1,
                                        path="../results/bqn/plots/forecasts/",
                                        filename="horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_test")
            evaluation.loc[index, "average"] = average_model.evaluate(x=sc_ens_test_f,
                                                                      y=sc_obs_test_f)
            print(evaluation)

            # Make model persistent for future evaluation
            average_model.save(
                "../results/bqn/models/horizon:" + str(horizon) + "_agg:" + aggregation + ".h5")

    evaluation.to_csv("../results/bqn/crps_evaluation.csv")


# This does not work since each model needs its own custom loss function which must be registered beforehand
def load_models() -> dict:
    models = {(horizon, aggregation): keras.models.load_model(
        "../results/bqn/models/horizon:" + str(horizon)
        + "_agg:" + aggregation + ".h5")
        for (horizon, aggregation) in product(horizons, aggregations)}
    return models


# Instead load each model individually
def load_model(horizon, aggregation, crps) -> keras.Model:
    return keras.models.load_model(
        "../results/bqn/models/horizon:" + str(horizon) + "_agg:" + aggregation + ".h5",
        custom_objects={"crps": crps})


def plot_rank_histograms_and_forecasts(plots_path=None):
    hps = load_hyperparameters_from_folders("../results/bqn/hps/")
    fixed_params = {"variables": variables, "train_split": 0.85, "data_set": "sweden"}
    for horizon in horizons:
        fixed_params["horizon"] = horizon
        # Import data
        sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers \
            = preprocess_data(horizon=horizon, train_variables=fixed_params["variables"],
                              train_split=fixed_params["train_split"],
                              data_set=fixed_params["data_set"])
        obs_scaler = output_scalers["wind_power"]
        obs_max = obs_scaler.data_max_
        obs_min = obs_scaler.data_min_

        for aggregation in aggregations:
            index = (horizon, aggregation)
            degree = hps.loc[index, "degree"]
            crps = build_crps_loss3(degree, obs_min, obs_max)
            model = load_model(horizon, aggregation, crps)
            fixed_params["aggregation"] = aggregation

            (sc_ens_train_f,
             sc_ens_test_f,
             sc_obs_train_f,
             sc_obs_test_f) = format_data(sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test,
                                          fixed_params["aggregation"], fixed_params["data_set"])
            train = pd.DataFrame(model.predict(sc_ens_train_f), index=sc_ens_train_f.index)
            test = pd.DataFrame(model.predict(sc_ens_test_f), index=sc_obs_test_f.index)
            with plt.xkcd():
                generate_histogram_plot(sc_obs_train_f,
                                        train,
                                        "Rank Histogram - Train data\n Horizon " + str(horizon) +
                                        " - Aggregation " + aggregation,
                                        21,
                                        path=plots_path + "rankhistograms/horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_train.png" if plots_path is not None else None
                                        )
                generate_histogram_plot(sc_obs_test_f,
                                        test,
                                        "Rank Histogram - Test data\n Horizon " + str(horizon) +
                                        " - Aggregation " + aggregation,
                                        21,
                                        path=plots_path + "rankhistograms/horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_test.png" if plots_path is not None else None
                                        )
                generate_forecast_plots(sc_obs_test_f[::51], test[::51],
                                        name="Example Forecast Plot",
                                        quantile_levels=np.arange(0, 1.01, 0.01), n=1,
                                        path=plots_path + "forecasts/horizon:" + str(
                                            horizon) + "_agg:" + aggregation + ".png" if plots_path is not None else None
                                        )


def analyze_first_coefficient(plots_path=None):
    hps = load_hyperparameters_from_folders("../results/bqn/hps/")
    fixed_params = {"variables": variables, "train_split": 0.85, "data_set": "sweden"}
    coeffs = pd.DataFrame(index=horizons, columns=aggregations)
    for horizon in horizons:
        fixed_params["horizon"] = horizon
        # Import data
        sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers \
            = preprocess_data(horizon=horizon, train_variables=fixed_params["variables"],
                              train_split=fixed_params["train_split"],
                              data_set=fixed_params["data_set"])
        obs_scaler = output_scalers["wind_power"]
        obs_max = obs_scaler.data_max_
        obs_min = obs_scaler.data_min_

        for aggregation in aggregations:
            index = (horizon, aggregation)
            degree = hps.loc[index, "degree"]
            crps = build_crps_loss3(degree, obs_min, obs_max)
            model = load_model(horizon, aggregation, crps)
            fixed_params["aggregation"] = aggregation

            (sc_ens_train_f,
             sc_ens_test_f,
             sc_obs_train_f,
             sc_obs_test_f) = format_data(sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test,
                                          fixed_params["aggregation"], fixed_params["data_set"])
            test = pd.DataFrame(model.predict(sc_ens_test_f), index=sc_obs_test_f.index)
            coeffs.loc[index] = test[0].mean()
    with plt.xkcd():
        coeffs.plot(figsize=figsize)
        plt.xticks(horizons)
        plt.xlabel("Horizon", fontdict=fontdict_axis)
        plt.title("Average first coefficients", fontdict=fontdict_title)
        if plots_path is None:
            plt.show()
        else:
            plt.savefig(plots_path + "first_coeffs.png")


if __name__ == "__main__":
    evaluation_path = "../results/bqn/crps_evaluation.csv"
    plots_path = "../results/bqn/plots/"
    # evaluate_best_hps()
    # plot_rank_histograms_and_forecasts()
    # plot_crps_per_horizon_per_aggregation(evaluation_path, plots_path)
    # plot_crps_per_horizon(evaluation_path, plots_path)
    # plot_crps_per_aggregation(evaluation_path, plots_path)
    # analyze_first_coefficient(plots_path)
