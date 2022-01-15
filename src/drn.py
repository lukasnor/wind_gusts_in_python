# Imports
import datetime
import keras.losses
import pandas as pd
import tensorflow as tf
from keras import layers, Model
import numpy as np
import tensorflow_probability as tfp
from keras.callbacks import EarlyStopping

#  Import standard logistic distribution for the custom loss
tfd = tfp.distributions.Logistic(0, 1)


# Custom loss function
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
    p_0 = tfd.cdf(z_0)  # F(z_0)
    lp_0 = tfd.log_cdf(z_0)  # log( F(z_0) )
    p_m0 = tfd.cdf(-z_0)  # 1 - F(z_0) = F(-z_0)
    lp_m0 = tfd.log_cdf(-z_0)  # log( 1 - F(z_0)) = log( F(-z_0) )
    lp_my = tfd.log_cdf(-z_y)  # log( 1 - F(z_y)) = log( F(-z_y) )

    # Calculate sigma
    b = lp_m0 - (1 + 2 * lp_my) / p_m0 - tf.square(p_0) * lp_0 / tf.square(p_m0)

    # Calculate CRPS
    res = tf.abs(z - y_true) - (z - mu) * (1 + p_0) / p_m0 + sigma * b

    # Calculate mean
    res = tf.math.reduce_mean(res)

    # Return mean CRPS
    return res


# Return the model of Benedikt Schulz for estimating location and scale parameters
def get_benedikts_model(n_dir_preds, n_loc, emb_dim, lay1, actv):
    # Inputs
    id_input = layers.Input(shape=1, name="id_input")  # Maybe names has to be changed to be compatible with pywatts
    dir_input = layers.Input(shape=n_dir_preds, name="dir_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc + 1, output_dim=emb_dim, input_length=1)(id_input)
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


# Test code to see if model with custom loss works
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


# Forecast from offshore_ensembles_horizon12 -> offshore_observations[speed]

# Import data
observations: pd.DataFrame = pd.read_csv("data/Offshore_Observations.csv", index_col=0)  # time is index
observations.index = pd.to_datetime(observations.index)  # convert Index to DateTimeIndex
pred_vars = observations.keys().drop(["horizon", "is_origin"])  # Index(['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'speed'])
observations = observations[pred_vars]  # Leave out "horizon" and "is_origin" from Observations

ensembles: pd.DataFrame = pd.read_csv("data/Offshore_Ensembles.csv")
ensembles = ensembles.pivot(index=["horizon", "time"], columns=["number"], values=pred_vars)
ensembles = ensembles.loc[12, :]  # only use 12 hour horizon
ensembles.index = pd.to_datetime(ensembles.index)  # convert Index to DateTimeIndex
observations = observations.loc[ensembles.index]  # only use the observations corresponding to the forecasts

# Plan: Daten nach Train und Test splitten, Pipeline aufbauen, laufen lassen, Calibration anschauen
# Loss: Nehme Verteilung für wind speed an und nehme CRPS dafür
# Netz: -Zuerst nur mit wind speed Werten aller Ensembles arbeiten
#       -später mit allen Spalten arbeiten, dann diese vielleicht zuerst aggregieren?