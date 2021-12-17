import pandas as pd
from keras import layers, Model
import numpy as np
import tensorflow_probability as tfp

# Import standard logistic distribution for the custom loss
tfd = tfp.distributions.Logistic(1, 0)


# Custom loss function
def custom_loss(y_true, y_pred):
    # Get location and scale
    mu = np.dot(y_pred, np.array([[1], [0]]))
    sigma = np.dot(y_pred, np.array([[0], [1]]))

    # Truncated in zero
    z = np.maximum(0, y_true)

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
    b = lp_m0 - (1 + 2 * lp_my) / p_m0 - np.square(p_0) * lp_0 / np.square(p_m0)

    # Calculate CRPS
    res = np.abs(z - y_true) - (z - mu) * (1 + p_0) / p_m0 + sigma * b

    # Calculate mean
    res = np.mean(res)

    # Return mean CRPS
    return (res)


def get_keras_model(n_dir_preds, n_loc, emb_dim, lay1, actv):
    # Inputs
    id_input = layers.Input(shape=1, name="id_input")  # Maybe names has to be changed to be compatible with pywatts
    dir_input = layers.Input(shape=n_dir_preds, name="dir_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc + 1, output_dim=emb_dim, input_length=1)(id_input)

    # Merge Inputs
    merged = layers.Concatenate()([dir_input, station_embedding_part])

    # Hidden Layers
    hidden1 = layers.Dense(units=lay1, activation=actv)(merged)
    hidden2 = layers.Dense(units=lay1 / 2, activation=actv)(hidden1)

    # Outputs
    loc_out = layers.Dense(units=1, activation="softplus")(hidden2)
    scale_out = layers.Dense(units=1, activation="softplus")(hidden2)

    output = layers.Concatenate()([loc_out, scale_out])

    model = Model(inputs=[id_input, dir_input], outputs=output)

    return model


if __name__ == '__main__':
    # Import data
    train = pd.read_csv("data/df_train.csv", index_col=0)
    test = pd.read_csv("data/df_test.csv", index_col=0)

    # Preprocessing pipeline

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
    # Split data into training and validation

    # Scale dir_preds w/o location