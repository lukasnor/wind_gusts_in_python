import json
from itertools import product
import keras.optimizer_v2.adam
import keras_tuner
from keras.callbacks import EarlyStopping
from keras_tuner import Hyperband
from pandas import DataFrame

from preprocessing import import_data, scale_data, format_data, categorify_data
from hen import binning_scheme, get_model, build_wrapped_crossentropy_loss, build_wrapped_hen_crps,\
    vincentize_forecasts, evaluation_crps

# possible values of the fixed params
horizons = [3, 6, 9, 12, 15, 18, 21, 24]
# horizons = [3]  # For test purposes
variables = ["u100", "v100", "t2m", "sp", "speed", "wind_power"]
variable_selections = [variables]
aggregations = ["single", "single+std", "mean+std", "mean", "all"]
# aggregations = ["mean+std"]  # For test purposes
fixed_params_selections = [{"horizon": a, "variables": b, "aggregation": c} for a, b, c in
                           product(horizons, variable_selections, aggregations)]

def run_tuner():
    for horizon in horizons:
        fixed_params = {"horizon": horizon,
                        "variables": variables,
                        "train_split": 0.85,
                        "n_bins": 20}  # I don't know how to make the number of bins a hyperparameter, since the data (no. of columns) depends on it

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
            fixed_params["aggregation"] = aggregation

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

            if aggregation in ["single", "single+std"]:
                fixed_params["batch_size"] = 500
            else:
                fixed_params["batch_size"] = 25

            def model_builder(hp):
                hp.Fixed(name="horizon", value=horizon)
                hp.Fixed(name="aggregation", value=aggregation)
                hp_n_bins = hp.Fixed(name="n_bins", value=fixed_params["n_bins"])
                hp_input_size = hp.Fixed(name="input_size", value=len(sc_ens_train_fc.columns))
                hp.Fixed(name="batch_size", value=fixed_params["batch_size"])

                max_depth = 5
                hp_depth = hp.Int(name="depth", min_value=2, max_value=max_depth, default=max_depth)

                min_layer_size = hp_n_bins
                max_layer_size = min_layer_size * 3
                layer_step = int(min_layer_size / 2)
                hp_layer_sizes = [
                    hp.Int(name="layer" + str(i) + "_size", min_value=min_layer_size,
                           max_value=max_layer_size, default=min_layer_size,
                           step=layer_step, parent_name="depth",
                           parent_values=[*range(i + 1, max_depth + 1)]) for i in
                    range(max_depth)]
                for _ in range(hp_layer_sizes.count(None)):
                    hp_layer_sizes.remove(None)
                hp_layer_sizes.sort(reverse=True)

                hp_learning_rate = hp.Choice(name="learning_rate",
                                             values=[1e-1, 1e-2, 1e-3, 1e-4])
                hp_activation = hp.Choice(name="activation", values=["relu", "selu"])
                hp_activations = [hp_activation for _ in range(hp_depth)]

                model = get_model(name="foo",
                                  input_size=hp_input_size,
                                  layer_sizes=hp_layer_sizes,
                                  activations=hp_activations,
                                  n_bins=hp_n_bins)
                model.compile(loss=build_wrapped_crossentropy_loss(),
                              optimizer=keras.optimizer_v2.adam.Adam(hp_learning_rate),
                              metrics=[build_wrapped_hen_crps(bin_edges)])
                return model

            tuner = Hyperband(model_builder,
                              objective=keras_tuner.Objective('val_crps', direction="min"),
                              max_epochs=300,
                              factor=3,
                              directory='../results/hen/tuning',
                              project_name="horizon:" + str(horizon) + "_agg:" + str(aggregation))
            stop_early = EarlyStopping(monitor='val_crps', patience=25, restore_best_weights=True)

            # Run the search
            tuner.search(x=sc_ens_train_fc,
                         y=sc_obs_train_fc,
                         epochs=300,
                         batch_size=fixed_params["batch_size"],
                         validation_split=0.1,
                         callbacks=[stop_early],
                         use_multiprocessing=True)

            # Get the three optimal hyperparameter sets, and compare them
            best_hps_candidates = tuner.get_best_hyperparameters(num_trials=3)
            evaluations = []
            # For each set of hps, average model over 10 runs
            for hp in best_hps_candidates:
                n_runs = 10
                print(hp.values)
                models = [get_model(name="model" + str(i),
                                    input_size=hp["input_size"],
                                    layer_sizes=[hp["layer" + str(i) + "_size"] for i in
                                                 range(hp["depth"])],
                                    activations=[hp["activation"] for _ in range(hp["depth"])],
                                    n_bins=hp["n_bins"])
                          for i in range(n_runs)]
                bin_probs_list = []
                for model in models:
                    model.compile(optimizer=keras.optimizer_v2.adam.Adam(hp["learning_rate"]),
                                  loss=build_wrapped_crossentropy_loss(),
                                  metrics=[build_wrapped_hen_crps(bin_edges)])
                    print("Training of ", model.name)

                    # Train run of a model
                    model.fit(x=sc_ens_train_fc,
                              y=sc_obs_train_fc,
                              batch_size=hp["batch_size"],
                              epochs=300,
                              verbose=1,
                              validation_freq=1,
                              validation_split=0.1,
                              callbacks=[stop_early],
                              use_multiprocessing=True,
                              )

                    # Make forecast
                    bin_probs_forecast = DataFrame(index=sc_ens_test_fc.index,
                                                   data=model.predict(sc_ens_test_fc))

                    # Save prediction for Vincentization
                    bin_probs_list.append(bin_probs_forecast)

                # Evaluate runs together
                new_bin_edges, new_bin_probs = vincentize_forecasts(bin_edges, bin_probs_list, rounding=3)
                crps_value = evaluation_crps(sc_obs_test_f, new_bin_probs, new_bin_edges)
                evaluations.append(crps_value)

            # Find the best hps
            best_index = evaluations.index(min(evaluations))
            best_hps = (best_hps_candidates[best_index]).values

            print(best_hps)
            print(evaluations[best_index])

            # Save the best hps
            with open("../results/hen/hps/horizon:" + str(horizon) + "_agg:" + str(
                    aggregation) + ".json", "w") as file:
                json.dump(best_hps, file, indent=2)

if __name__ == "__main__":
    run_tuner()