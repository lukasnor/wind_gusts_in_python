# Imports
import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, Model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.initializers import initializers_v2 as inits
from src.preprocessing import preprocess_data, format_data

# For plot formatting
fontdict_title = {"fontweight": "bold", "fontsize": 24}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}
figsize = (13, 7)

#  Import standard logistic distribution for the custom loss
logistic = tfp.distributions.Logistic(0, 1)
tfd = tfp.distributions
Gamma = lambda x: tf.math.exp(tf.math.lgamma(x))


def custom_mae(y_true, y_pred):
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred[:, 0])


# Custom loss function / crps of a truncated logistic distribution, short version of the R
# code in scoringRules package
def crps_logistic_loss(y_true, y_pred):
    # Get location and scale
    # mu = np.dot(y_pred, np.array([[1], [0]]))
    mu = y_pred[0]
    # sigma = np.dot(y_pred, np.array([[0], [1]]))
    sigma = y_pred[1]
    # Truncated in zero
    z = tf.maximum(0.0, y_true)
    # Standardization
    z_0 = -mu / sigma  # q ~= -18.4 -> p = 1e-8
    z_y = (z - mu) / sigma

    # Calculate CDFs
    p_0 = logistic.cdf(z_0)  # F(z_0)
    lp_0 = logistic.log_cdf(z_0)  # log( F(z_0) )
    p_m0 = logistic.cdf(-z_0)  # 1 - F(z_0) = F(-z_0)
    lp_m0 = logistic.log_cdf(-z_0)  # log( 1 - F(z_0)) = log( F(-z_0) )
    lp_my = logistic.log_cdf(-z_y)  # log( 1 - F(z_y)) = log( F(-z_y) )

    # Calculate sigma
    b = lp_m0 - (1 + 2 * lp_my) / p_m0 - tf.square(p_0) * lp_0 / tf.square(p_m0)

    # Calculate CRPS
    res = tf.abs(z - y_true) - (z - mu) * (1 + p_0) / p_m0 + sigma * b

    # Calculate mean
    res = tf.math.reduce_mean(res)

    # Return mean CRPS
    return res


# Custom loss function / crps of a gamma distribution
# y_pred being the shape and the rate parameter
def crps_gamma_loss(y_true, y_pred):
    alpha = tf.squeeze(y_pred[:, 0])
    beta = tf.squeeze(y_pred[:, 1])
    dist1 = tfp.distributions.Gamma(alpha, beta)
    dist2 = tfp.distributions.Gamma(alpha + 1, beta)
    s1 = y_true * (2 * dist1.cdf(y_true) - 1)
    s2 = alpha / beta * (2 * dist2.cdf(y_true) - 1)
    B = Gamma(0.5) * Gamma(alpha) / Gamma(alpha + 0.5)
    s3 = 1 / (beta * B)
    return s1 - s2 - s3


# Return the model of Benedikt Schulz for estimating location and scale parameters
def get_benedikts_model(n_dir_preds, n_loc, emb_dim, lay1, actv):
    # Inputs
    id_input = layers.Input(shape=1,
                            name="id_input")  # Maybe names has to be changed to be compatible with pywatts
    dir_input = layers.Input(shape=n_dir_preds, name="dir_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc + 1, output_dim=emb_dim,
                                              input_length=1)(id_input)
    station_embedding_flat = layers.Flatten()(station_embedding_part)
    # Merge Inputs
    merged = layers.Concatenate()([dir_input, station_embedding_flat])

    # Hidden Layers
    hidden1 = layers.Dense(units=lay1, activation=actv)(merged)
    hidden2 = layers.Dense(units=lay1 / 2, activation=actv)(hidden1)

    # Outputs
    loc_out = layers.Dense(units=1, activation="softplus")(hidden2)
    scale_out = layers.Dense(units=1, activation="softplus")(hidden2)
    output = layers.Concatenate()([loc_out, scale_out])

    # Model
    model = Model(inputs=[id_input, dir_input], outputs=output)

    return model


# A simple aggregation model to summarize ensemble member info wrt to one variable
def get_aggregation_model(name: str, n_ens, width: int, activations):
    initializer_dict = {"kernel_initializer": inits.GlorotNormal(seed=42),
                        "bias_initializer": inits.Zeros,
                        "kernel_regularizer": "l1"
                        }
    # initializer_dict = {"bias_initializer": inits.RandomNormal(mean=0.05, seed=42)}
    initializer_dict = {}
    input = layers.Input(shape=n_ens, name="input")
    hidden1 = layers.Dense(name="hidden1", units=width, activation=activations[0],
                           **initializer_dict)(input)
    # hidden2 = layers.Dense(name="hidden2", units=width, activation=activations[1], **initializer_dict)(hidden1)
    output = layers.Dense(units=1)(hidden1)
    model = Model(name=name, inputs=input, outputs=output)
    return model


def get_drn_model(name: str, input_size: int, layer_sizes: [int], activations: [str]) -> Model:
    # Inputs
    input = layers.Input(shape=input_size, name="input")

    # Hidden layers
    x = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden0")(input)
    for i in range(len(layer_sizes) - 1):
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1],
                         name="hidden" + str(i + 1))(x)

    # Output, size 2 for location and scale
    output = layers.Dense(name="output", units=2, activation="softplus")(
        x)  # smooth, non-negative and proportional

    # Model
    return Model(name=name, inputs=input, outputs=output)


def get_base_logistic_model(name: str, input_size: int, layer_sizes: [int],
                            activations: [str]) -> Model:
    # Inputs
    input = layers.Input(shape=input_size, name="input")

    # Hidden layers
    x = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden0")(input)
    for i in range(len(layer_sizes) - 1):
        x = layers.Dense(units=layer_sizes[i + 1], activation=activations[i + 1],
                         name="hidden" + str(i + 1))(x)

    # Output, size 1 for location
    location = layers.Dense(name="output", units=1, activation="softplus")(
        x)  # smooth, non-negative and proportional
    output = layers.Concatenate()([location, 0.1 * tf.ones_like(location)])
    # Model
    return Model(name=name, inputs=input, outputs=output)


def generate_gamma_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame,
                                  name: str, n=None, path: str = None) -> None:
    quantiles = np.linspace(0, 1, 50)
    if n is None:
        n = y_true.shape[0]
    for i in range(n):
        plt.figure(figsize=figsize)
        plt.plot(scipy.stats.gamma.ppf(quantiles,
                                       (y_pred.iloc[i])[0],
                                       scale=1 / ((y_pred.iloc[i])[1])),
                 quantiles, color="blue", label="forecast")
        plt.vlines(y_true.iloc[i], ymin=0, ymax=1,
                   label="observation", color="red", linestyles="dashed")
        plt.xlim(left=0.0, right=max(1.0, plt.axis()[1]))
        plt.title(name + " - Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(path)


# path with a trailing '/'!
def generate_logistic_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame,
                                     name: str, n=None, path: str = None,
                                     filename: str = "forecast") -> None:
    x = np.linspace(0, 1.5, 150)
    loc = y_pred.values[:, 0]
    scale = y_pred.values[:, 1]
    if n is None:
        n = y_true.shape[0]
    for i in range(n):
        plt.figure(figsize=figsize)
        plt.plot(x, (logistic.cdf((x - loc[i]) / scale[i]) - logistic.cdf(-loc[i] / scale[i])) / (
                1 - logistic.cdf(-loc[i] / scale[i])), color="blue",
                 label="forecast")
        plt.vlines(y_true.iloc[i], ymin=0, ymax=1,
                   label="observation", color="red", linestyles="dashed")
        plt.xlim(left=0.0, right=np.maximum(1.0, y_true.values[i]))
        plt.title(name + " - Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        if path is None:
            plt.show()
        else:
            plt.savefig(path + filename + "_" + str(i) + ".png")


# Test code to see if benedikts model with custom loss works
def try_out():
    # Import data
    train_data: pd.DataFrame = pd.read_csv("data/df_train.csv", index_col=0)
    test_data: pd.DataFrame = pd.read_csv("data/df_test.csv", index_col=0)

    # Define Hyperparameters
    hpar_ls = {"n_sim": 10,
               "lr_adam": 5e-4,  # previously 1e-3
               "n_epochs": 150,
               "n_patience": 10,
               "n_batch": 64,
               "emb_dim": 10,
               "lay1": 64,
               "actv": "softplus",
               "nn_verbose": 0}

    # Data preparation

    # Remove unnecessary columns, so no individual ensemble members
    train_vars = train_data.keys().drop(["ens_" + str(i) for i in range(1, 21)])

    train_data = train_data.loc[:, train_vars]

    # Remove observations from training data
    test_vars = train_vars.drop(["obs"])
    test_data = test_data.loc[:, train_vars]

    # Number of direct predictants w.o. location
    dir_pred_vars = train_vars.drop(["location", "obs"])

    # Split data into training and validation
    # val_data = train_data[-10:] # not necessary with validation_split in model.fit()

    # Test model right here
    model = get_benedikts_model(n_dir_preds=len(dir_pred_vars),
                                n_loc=len(train_data["location"].unique()),
                                emb_dim=hpar_ls["emb_dim"],
                                lay1=hpar_ls["lay1"],
                                actv=hpar_ls["actv"])
    model.compile(optimizer="adam", loss=crps_logistic_loss)  # vielleicht anderen Loss probieren
    # model.compile(optimizer="adam",loss=keras.losses.mean_squared_error)
    # Compile the model
    history = model.fit([train_data["location"],
                         train_data[dir_pred_vars]],
                        train_data["obs"],
                        batch_size=hpar_ls["n_batch"],
                        epochs=hpar_ls["n_epochs"],
                        validation_split=0.1,
                        callbacks=EarlyStopping(patience=hpar_ls["n_patience"]))


# Try to construct a model, that beats the mean when aggregating the ensemble weather variables
def try_predict_each_variable():
    # Import data
    observations: pd.DataFrame = pd.read_csv("../data/Offshore_Observations.csv",
                                             index_col=0)  # time is index
    observations.index = pd.to_datetime(observations.index)  # convert Index to DateTimeIndex
    pred_vars = observations.keys().drop(
        ["horizon", "is_origin"])  # Index(['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'speed'])
    observations = observations[pred_vars]  # leave out "horizon" and "is_origin" from observations
    observations = observations.sort_index(level=0)

    ensembles: pd.DataFrame = pd.read_csv("../data/Offshore_Ensembles.csv")
    ensembles["time"] = pd.to_datetime(ensembles["time"],
                                       infer_datetime_format=True)  # convert time column to datetime
    ensembles = ensembles.pivot(index=["horizon", "time", "number"],
                                columns=[])  # create multiindex
    ensembles = ensembles[pred_vars]  # reduce columns to necessary ones
    ensembles = ensembles.sort_index(
        level=[0, 1,
               2])  # sort by horizon first (somewhat irrelevant), then by date (relevant for iloc!)

    horizon = 6
    ensembles = ensembles.loc[horizon]  # select horizon from data
    observations = observations.loc[
        ensembles.index.get_level_values(
            0).unique()]  # only use the observations corresponding to the forecasts

    n_obs = len(observations)  # 577 for each horizon
    n_ens = ensembles.index.levshape[1]
    split = 0.85
    n_train_split = int(split * n_obs)  # number of dates

    i_train = np.sort(np.random.choice(n_obs, size=n_train_split,
                                       replace=False))  # randomize the train and test set
    i_test = np.delete(np.array(range(n_obs)), i_train)
    # i_train = np.array(range(n_train_split))  # unrandomized train and test set
    # i_test = np.array(range(n_train_split, n_obs))
    dates_train = observations.index[i_train]
    dates_test = observations.index[i_test]
    train = pd.DataFrame(ensembles.loc[dates_train])
    test = pd.DataFrame(ensembles.loc[dates_test])

    # Normalize Data
    scaler = StandardScaler()
    scaler.fit(train)
    train_norm = pd.DataFrame(data=scaler.transform(train), index=train.index,
                              columns=train.columns)
    test_norm = pd.DataFrame(data=scaler.transform(test), index=test.index, columns=test.columns)
    observations_norm = pd.DataFrame(data=scaler.transform(observations), index=observations.index,
                                     columns=observations.columns)

    # mean model as reference
    input = layers.Input(name="input", shape=n_ens)
    mean_model = Model(name="mean_model", inputs=input,
                       outputs=layers.Lambda(name="mean_layer",
                                             function=(lambda ens: tf.reduce_mean(ens, axis=1)),
                                             output_shape=1)(input))
    mean_model.trainable = False
    mean_model.compile(optimizer="adam", loss='mean_absolute_error')

    # And go
    models = []
    bar_data = []
    pred_vars = pred_vars.drop(["msl"])  # "msl" column messed up in data
    # pred_vars = ["d2m" for _ in range(5)] #+ ["t2m" for _ in range(5)]
    # pred_vars = ["speed"]
    for var_name in pred_vars:
        train = train_norm.reset_index().pivot(index="time", columns="number", values=var_name)
        test = test_norm.reset_index().pivot(index="time", columns="number", values=var_name)

        # main model
        model: Model = get_aggregation_model(name=var_name + "_model", n_ens=n_ens, width=15,
                                             activations=["selu", "selu"])
        model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.001),
                      # optimizer="adam",
                      loss='mean_absolute_error'
                      )
        print("Training of ", model.name)
        history = model.fit(y=observations_norm[var_name].iloc[i_train],
                            x=train,
                            batch_size=20,
                            epochs=200,
                            verbose=1,
                            validation_freq=1,
                            validation_split=0.15,
                            callbacks=[EarlyStopping(patience=50, restore_best_weights=True)],
                            use_multiprocessing=True
                            )
        print("Evaluation of ", model.name)
        evaluation = model.evaluate(x=test,
                                    y=observations_norm[var_name].iloc[i_test]
                                    )
        models.append(model)

        plt.subplots(figsize=(12, 8))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Learning curve of " + var_name, fontweight="bold", fontsize=18)
        plt.show()

        plt.subplots(figsize=(12, 8))
        plt.plot(pd.DataFrame(data=model.predict(train), index=train.index), label="Model")
        plt.plot(pd.DataFrame(data=mean_model(train), index=train.index), label="Mean")
        plt.plot(observations_norm[var_name].iloc[i_train], label="Observation")
        plt.legend()
        plt.title("Train set of " + var_name, fontweight="bold", fontsize=18)
        plt.show()

        # Mean Model as reference
        print("Base Value:")
        mean_evaluation = mean_model.evaluate(x=test,
                                              y=observations_norm[var_name].iloc[i_test]
                                              )
        print()

        fig = plt.subplots(figsize=(12, 8))
        plt.plot(pd.DataFrame(data=model.predict(test), index=test.index), label="Model")
        plt.plot(pd.DataFrame(data=mean_model.predict(test), index=test.index), label="Mean")
        plt.plot(observations_norm[var_name].iloc[i_test], label="Observation")
        plt.title("Test set of " + var_name, fontweight="bold", fontsize=18)
        plt.legend()
        plt.show()

        bar_data.append([evaluation, mean_evaluation])

    # Generate bar plot
    bar_df = pd.DataFrame(data=np.array(bar_data), index=pred_vars, columns=["agg", "mean"])
    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(bar_data))
    br2 = [x + barWidth for x in br1]
    # Make the plot
    plt.bar(br1, bar_df["agg"], color='r', width=barWidth,
            edgecolor='grey', label='agg')
    plt.bar(br2, bar_df["mean"], color='g', width=barWidth,
            edgecolor='grey', label='mean')
    # Adding Xticks
    plt.xlabel('Variable', fontweight='bold', fontsize=15)
    plt.ylabel('Evaluation', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth / 2 for r in range(len(bar_df))],
               pred_vars)
    plt.legend()
    plt.title("Loss Comparison", fontweight="bold", fontsize=18)
    plt.show()


if __name__ == "__main__":

    # Get the data in a processed form
    h_pars = {"horizon": 3,  #
              "variables": None,
              "train_split": 0.85,

              "aggregation": "mean+std",
              "layer_sizes": [20, 15],
              "activations": ["selu", "selu", "selu"],

              "batch_size": 25,
              "patience": 27,
              }
    # Default value for activation is "selu" if activations do not match layer_sizes
    if h_pars["activations"] is None or \
            not len(h_pars["activations"]) == len(h_pars["layer_sizes"]):
        h_pars["activations"] = ["selu" for i in range(len(h_pars["layer_sizes"]))]
    # Default value for variables is 'using all variables'
    if h_pars["variables"] is None:
        h_pars["variables"] = ["u100", "v100", "t2m", "sp", "speed"]

    # Import the data
    sc_ens_train, sc_ens_test, sc_obs_train, sc_obs_test, input_scalers, output_scalers \
        = preprocess_data(h_pars["horizon"], h_pars["variables"], h_pars["train_split"])
    obs_scaler = output_scalers["wind_power"]
    obs_max = obs_scaler.data_max_
    obs_min = obs_scaler.data_min_

    # Format the data
    sc_ens_train_f, sc_ens_test_f, sc_obs_train_f, sc_obs_test_f = format_data(sc_ens_train,
                                                                               sc_ens_test,
                                                                               sc_obs_train,
                                                                               sc_obs_test,
                                                                               h_pars["aggregation"])
    # Build model
    base_model = get_base_logistic_model(name="base_model",
                                         input_size=len(sc_ens_train_f.columns),
                                         layer_sizes=h_pars["layer_sizes"],
                                         activations=h_pars["activations"])
    base_model.compile(optimizer="adam", loss=custom_mae)
    drn_model = get_drn_model(name="drn_model",
                              input_size=len(sc_ens_train_f.columns),
                              layer_sizes=h_pars["layer_sizes"],
                              activations=h_pars["activations"])
    drn_model.compile(optimizer="adam", loss=crps_logistic_loss)
    # Fit model
    base_model.fit(x=sc_ens_train_f,
                   y=sc_obs_train_f,
                   batch_size=h_pars["batch_size"],
                   epochs=100,
                   verbose=1,
                   validation_freq=1,
                   validation_split=0.1,
                   callbacks=[EarlyStopping(monitor="val_loss",
                                            patience=h_pars["patience"],
                                            restore_best_weights=True)],
                   use_multiprocessing=True
                   )
    drn_model.fit(x=sc_ens_train_f,
                  y=sc_obs_train_f,
                  batch_size=h_pars["batch_size"],
                  epochs=100,
                  verbose=1,
                  validation_freq=1,
                  validation_split=0.1,
                  callbacks=[EarlyStopping(monitor="val_loss",
                                           patience=h_pars["patience"],
                                           restore_best_weights=True)],
                  use_multiprocessing=True
                  )

    base_test = pd.DataFrame(base_model.predict(sc_ens_test_f), index=sc_ens_test_f.index)
    drn_test = pd.DataFrame(drn_model.predict(sc_ens_test_f), index=sc_ens_test_f.index)

    # Evaluation
    base_model.compile(optimizer="adam", loss=crps_logistic_loss)
    print("Base Model:", base_model.evaluate(sc_ens_test_f, sc_obs_test_f))
    print("DRN Model:", drn_model.evaluate(sc_ens_test_f, sc_obs_test_f))

    path = "../results/drn/plots/forecasts/"
    # Plots
    with plt.xkcd():
        generate_logistic_forecast_plots(sc_obs_test_f, base_test, "Base", n=10, path=None,
                                         filename="base_forecast")
        generate_logistic_forecast_plots(sc_obs_test_f, drn_test, "DRN", n=10, path=None,
                                         filename="forecast")
