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
            hp.Fixed(name="horizon", value=horizon)
            hp.Fixed(name="aggregation", value=aggregation)
            hp_input_size = hp.Fixed(name="input_size", value=len(sc_ens_train_f.columns))

            hp_degree = hp.Int(name="degree", min_value=4, max_value=20)

            max_depth = 5
            min_layer_size = max(int(hp_input_size / 4), hp.get("degree"))
            layer_step = max(int((hp_input_size / 2)), 1)
            hp_depth = hp.Int(name="depth", min_value=1, max_value=max_depth)
            for i in range(hp_depth):
                if i == 0:
                    hp.Int(name="layer0_size", min_value=min_layer_size, max_value=hp_input_size * 4,
                           step=layer_step)
                else:
                    hp.Int(name="layer" + str(i) + "_size", min_value=min_layer_size,
                           max_value=hp.get("layer" + str(i - 1) + "_size"), step=layer_step,
                           parent_name="depth", parent_values=[*range(i + 1, max_depth + 1)])
            hp_layer_sizes = [hp.get("layer" + str(i) + "_size") for i in range(hp_depth)]

            hp_learning_rate = hp.Choice(name="learning_rate", values=[1e-2, 1e-3, 1e-4])

            hp_activation = hp.Choice(name="activation", values=["relu", "selu"])
            hp_activations = [hp_activation for _ in range(hp_depth)]
            print("degree: " + str(hp_degree), "depth: " + str(hp_depth), "layer_sizes: " + str(hp_layer_sizes),
                  "activation: " + str(hp_activation), sep="\n")
            model = get_model(name="foo", input_size=hp_input_size, layer_sizes=hp_layer_sizes,
                              activations=hp_activations, degree=hp_degree)
            model.compile(loss=build_quantile_loss(hp_degree), optimizer=keras.optimizer_v2.adam.Adam(hp_learning_rate))
            return model


        tuner = Hyperband(model_builder,
                          objective='val_loss',
                          max_epochs=81,
                          factor=3,
                          directory='../results/tuning',
                          project_name="horizon:" + str(horizon) + "_agg:" + str(aggregation))
        stop_early = EarlyStopping(monitor='val_loss', patience=10)

        # Run the search
        tuner.search(sc_ens_train_f, sc_obs_train_f, epochs=150, validation_split=0.2, callbacks=[stop_early],
                     use_multiprocessing=True, workers=3)

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
