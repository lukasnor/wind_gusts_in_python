from itertools import product

import numpy as np
import pandas as pd
import keras
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from pandas import DataFrame

from tuner_helper import load_hyperparameters_from_folders, plot_crps_per_horizon, \
    plot_crps_per_aggregation, plot_crps_per_horizon_per_aggregation
from preprocessing import import_data, scale_data, format_data, categorify_data
from hen import binning_scheme, get_model, build_wrapped_crossentropy_loss, build_wrapped_hen_crps, \
    build_hen_crps, vincentize_forecasts, generate_histogram_plot, evaluation_crps, \
    generate_forecast_plots

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


def evaluate_best_hps():
    # Average 10 models
    n_runs = 10

    # Import the best hps
    hps = load_hyperparameters_from_folders("../results/hen/hps/")
    evaluation = pd.DataFrame(index=hps.index,
                              columns=["run" + str(i + 1) for i in range(n_runs)] + ["average"])

    for horizon in horizons:

        fixed_params = {"horizon": horizon,
                        "variables": variables,
                        "train_split": 0.85,
                        "n_bins": 25}

        # Import the data
        ens_train, \
        ens_test, \
        obs_train, \
        obs_test = import_data(horizon=fixed_params["horizon"],
                               variables=fixed_params["variables"],
                               train_split=fixed_params["train_split"])

        # Get the bin edges
        bin_edges = binning_scheme(obs_train, fixed_params["n_bins"])

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
            fixed_params["learning_rate"] = hps.loc[index, "learning_rate"]

            # Format input
            sc_ens_train_f, \
            sc_ens_test_f, \
            sc_obs_train_f, \
            sc_obs_test_f = format_data(sc_ens_train,
                                        sc_ens_test,
                                        sc_obs_train,
                                        sc_obs_test,
                                        aggregation=fixed_params["aggregation"])

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
            sc_ens_train_fc = sc_ens_train_fc.iloc[:, : (-1) * fixed_params["n_bins"]]
            sc_ens_test_fc = sc_ens_test_fc.iloc[:, : (-1) * fixed_params["n_bins"]]

            evaluations_train = []
            evaluations_test = []
            bin_probs_list_train = []
            bin_probs_list_test = []
            for i in range(n_runs):
                model = get_model(name="run" + str(i + 1),
                                  input_size=len(sc_ens_train_fc.columns),
                                  layer_sizes=fixed_params["layer_sizes"],
                                  activations=fixed_params["activations"],
                                  n_bins=fixed_params["n_bins"]
                                  )
                model.compile(loss=build_wrapped_crossentropy_loss(),
                              optimizer=keras.optimizer_v2.adam.Adam(fixed_params["learning_rate"]),
                              metrics=[build_wrapped_hen_crps(bin_edges)])

                # Train model
                model.fit(x=sc_ens_train_fc,
                          y=sc_obs_train_fc,
                          batch_size=fixed_params["batch_size"],
                          epochs=200,
                          verbose=1,
                          validation_freq=1,
                          validation_split=0.1,
                          callbacks=[EarlyStopping(monitor="val_loss",
                                                   patience=fixed_params["patience"],
                                                   restore_best_weights=True
                                                   )],
                          use_multiprocessing=True
                          )

                # Make forecasts
                bin_probs_train = DataFrame(index=sc_ens_train_fc.index,
                                            data=model.predict(sc_ens_train_fc))
                bin_probs_test = DataFrame(index=sc_ens_test_fc.index,
                                           data=model.predict(sc_ens_test_fc))

                # Save predictions for Vincentization
                bin_probs_list_train.append(bin_probs_train)
                bin_probs_list_test.append(bin_probs_test)

                # Evaluation
                crps_func = build_hen_crps(bin_edges)
                crps_val_test = crps_func(sc_obs_test_f.values, bin_probs_test.values).numpy()
                evaluation.loc[index, "run" + str(i + 1)] = crps_val_test
                evaluations_test.append(crps_val_test)
                print(evaluation)

                # Save the train evaluation for the weighted vincentization
                evaluations_train.append(crps_func(sc_obs_train_f.values,
                                                   bin_probs_train.values).numpy()
                                         )

                # Make model persistent
                model.save(
                    "../results/hen/models/horizon:" + str(
                        horizon) + "_agg:" + aggregation + "_" + str(i) + ".h5")

            # Evaluate the aggregated model
            weights_train = (1 / np.linspace(1, len(bin_probs_list_train), len(bin_probs_list_train)))[
                np.array(evaluations_train).argsort()]  # use a 'Benfordian' weight vector
            new_bin_edges_train, new_bin_probs_train = vincentize_forecasts(bin_edges,
                                                                            bin_probs_list_train,
                                                                            rounding=3,
                                                                            weights=weights_train)
            weights_test = \
            (1 / np.linspace(1, len(bin_probs_list_test), len(bin_probs_list_test)))[
                np.array(evaluations_test).argsort()]  # use a 'Benfordian' weight vector
            new_bin_edges_test, new_bin_probs_test = vincentize_forecasts(bin_edges,
                                                                          bin_probs_list_test,
                                                                          rounding=3,
                                                                          weights=weights_test)

            evaluation.loc[index, "average"] = evaluation_crps(sc_obs_test_f,
                                                               new_bin_probs_test,
                                                               new_bin_edges_test)
            print(evaluation)

            # Generate plots for average_model
            # Forecast plots first
            generate_forecast_plots(obs=sc_obs_test_f[::50],
                                    bin_probs=new_bin_probs_test[::50],
                                    bin_edges=new_bin_edges_test[::50],
                                    name="Test - Horizon " + str(
                                        horizon) + " - Aggregation " + aggregation,
                                    n=5,
                                    path="../results/hen/plots/forecasts/",
                                    filename="horizon:" + str(
                                        horizon) + "_agg:" + aggregation + "_test"
                                    )
            # Histogram Plots second
            with plt.xkcd():
                name = "Rank Histogram - Train data\n" + "Horizon " + str(
                    fixed_params["horizon"]) + " - Aggregation " + fixed_params["aggregation"]
                generate_histogram_plot(obs=sc_obs_train_f,
                                        bin_probs=new_bin_probs_train,
                                        bin_edges=new_bin_edges_train,
                                        name=name,
                                        bins=21,
                                        path="../results/hen/plots/rankhistograms/",
                                        filename="horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_train"
                                        )
                name = "Rank Histogram - Test data\n" + "Horizon " + str(
                    fixed_params["horizon"]) + " - Aggregation " + fixed_params["aggregation"]
                generate_histogram_plot(obs=sc_obs_test_f,
                                        bin_probs=new_bin_probs_test,
                                        bin_edges=new_bin_edges_test,
                                        name=name,
                                        bins=21,
                                        path="../results/hen/plots/rankhistograms/",
                                        filename="horizon:" + str(
                                            horizon) + "_agg:" + aggregation + "_test"
                                        )

    evaluation.to_csv("../results/hen/crps_evaluation.csv")


if __name__ == "__main__":
    # All folders need to exist beforehand, otherwise saving plots won't work
    evaluation_path = "../results/hen/crps_evaluation.csv"
    plots_path = "../results/hen/plots/"
    # evaluate_best_hps()
    # plot_crps_per_horizon(evaluation_path=evaluation_path, plots_path=plots_path)
    # plot_crps_per_aggregation(evaluation_path=evaluation_path, plots_path=plots_path)
    # plot_crps_per_horizon_per_aggregation(evaluation_path=evaluation_path, plots_path=plots_path)
