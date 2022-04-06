import math
from difflib import restore

import keras.optimizer_v2.adam
from keras.callbacks import EarlyStopping
from bqn import preprocess_data, format_data, build_quantile_loss, get_model
from itertools import product, chain, combinations
from kerastuner import Hyperband


# Returns all sublists of list
def powerset(list):
    # Example: powerset([1,2,3]) --> [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
    return map(lambda t: [i for i in t], chain.from_iterable(combinations(list, r) for r in range(len(list) + 1)))


# possible values of the fixed params
horizons = [3, 6, 9, 12, 15, 18, 21, 24]
variables = ["u100", "v100", "t2m", "sp", "speed"]
variable_selections = [variables]  # or =  list(powerset(variables))[1:]
aggregations = ["single", "single+std", "mean", "all"]
fixed_params_selections = [{"horizon": a, "variables": b, "aggregation": c} for a, b, c in
                           product(horizons, variable_selections, aggregations)]

# Test run for horizon = 6 and aggregation = single+std
aggregations = ["single+std"]
horizons = [6]
for horizon in horizons:
    fixed_params = {"horizon": horizon, "variables": variables, "train_split": 0.85}
    # Import data
    sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test = preprocess_data(fixed_params)

    for aggregation in aggregations:
        fixed_params["aggregation"] = aggregation
        # Format data according to aggregation method
        sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f = format_data(fixed_params, sc_ens_train,
                                                                                   sc_ens_test, sc_obs_train,
                                                                                   sc_obs_test)


        def model_builder(hp):
            hp_horizon = hp.Fixed(name="horizon", value=horizon)
            hp_aggregation = hp.Fixed(name="aggregation", value=aggregation)

            # TODO: Batch sizes cannot be tuned
            # hp_batch_size = None
            # if hp_aggregation == "all" or hp_aggregation == "mean":
            #     hp_batch_size = hp.Int(name="batch_size", min_value=8, max_value=256, sampling="log", step=8,
            #                            parent_name="aggregation",
            #                            parent_values=["all", "mean"])
            # elif hp_aggregation == "single" or hp_aggregation == "single+std":
            #     hp_batch_size = hp.Int(name="batch_size", min_value=50, max_value=1000, sampling="log", step=50,
            #                            parent_name="aggregation",
            #                            parent_values=["single", "single+std"])

            hp_input_size = hp.Fixed(name="input_size", value=len(sc_ens_train_f.columns))

            hp_depth = hp.Int(name="depth", min_value=1, max_value=3)
            # TODO: What should the layer_sizes be?
            hp_layer_sizes = None
            if hp_aggregation == "all":
                hp_layer_sizes = hp.Choice(name="layer_sizes", values=["foo"])
            if hp_aggregation == "single" or hp_aggregation == "mean":
                hp_layer_sizes = hp.Choice(name="layer_sizes", values=["foo"])
            if hp_aggregation == "single+std":
                hp_layer_sizes = hp.Choice(name="layer_sizes", values=["foo"])
            # Dummy layer_sizes, all layers same size
            hp_layer_sizes = [
                hp.Int(name="layer" + str(i) + "_size", min_value=math.ceil(hp_input_size / 4),
                       max_value=hp_input_size * 4,
                       step=math.ceil(hp_input_size / 16)) for i in range(hp_depth)]
            hp_learning_rate = hp.Choice(name="learning_rate", values=[1e-2, 1e-3, 1e-4])
            hp_degree = hp.Int(name="degree", min_value=4, max_value=20)
            hp_activation = hp.Choice(name="activation", values=["relu", "selu"])
            hp_activations = [hp.Fixed(name="activations", value=hp_activation) for _ in range(hp_depth)]
            print("input_size: " + str(hp_input_size), "depth: " + str(hp_depth), "layer_sizes: " + str(hp_layer_sizes),
                  "activations: " + str(hp_activations), sep="\n")
            model = get_model(name="foo", input_size=hp_input_size, layer_sizes=hp_layer_sizes,
                              activations=hp_activations, degree=hp_degree)
            model.compile(loss=build_quantile_loss(hp_degree), optimizer=keras.optimizer_v2.adam.Adam(hp_learning_rate))
            return model


        tuner = Hyperband(model_builder,
                          objective='val_loss',
                          max_epochs=10,
                          factor=3,
                          directory='../results/tuning',
                          project_name="horizon:" + str(horizon) + "_agg:" + str(aggregation))
        stop_early = EarlyStopping(monitor='val_loss', patience=20)

        # Run the search
        tuner.search(sc_ens_train_f, sc_obs_train_f, epochs=150, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 200 epochs
        # TODO: Is that the sensible thing to do, if training runs differ alot? Maybe average over multiple runs later
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(sc_ens_train_f, sc_obs_train_f, epochs=200, validation_split=0.2,
                            callbacks=[EarlyStopping(patience=50, restore_best_weights=True)])

        # Figure out best epoch
        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(max(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # Retrain the model
        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(sc_ens_train_f, sc_obs_train_f, epochs=best_epoch, validation_split=0.2)

        # Evaluate the model
        eval_result = hypermodel.evaluate(sc_ens_test_f, sc_obs_test_f)
        print("test loss:", eval_result)
