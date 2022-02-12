import numpy as np
from scipy.special import binom
from keras import Model, layers
import tensorflow as tf


# Construction of the model
def get_model(name: str, n_ens_input: int, n_loc: int, n_emb: int, layer_sizes: [int], activations,
              degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=n_ens_input, name="ens_input")
    emb_input = layers.Input(shape=1, name="emb_input")

    # Embedding
    station_embedding_part = layers.Embedding(input_dim=n_loc, output_dim=n_emb, input_shape=1)(emb_input)
    station_embedding_flat = layers.Flatten()(station_embedding_part)
    # Merge Inputs
    merged = layers.Concatenate(name="merged")([ens_input, station_embedding_flat])

    # Hidden layers
    hidden1 = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden1")(merged)
    hidden2 = layers.Dense(units=layer_sizes[1], activation=activations[1], name="hidden2")(hidden1)

    # Output
    output = layers.Dense(units=degree + 1, activation="softplus")(hidden2)  # smooth, non-negative and proportional

    # Model
    model = Model(name="name", inputs=[ens_input, emb_input], outputs=output)
    return model


# Construction of the loss function
degree = 6
loss_quantile_levels = np.arange(0.01, 1, 0.01)  # 1% to 99% quantile levels for the loss, equidistant
lql = tf.constant(loss_quantile_levels)
bernsteins = np.array(
    [binom(degree, j) * np.power(loss_quantile_levels, j) * np.power(1 - loss_quantile_levels, degree - j) for j in
     range(degree + 1)]).transpose()
B = tf.constant(bernsteins)


# Multi-quantile loss: sum over all quantile losses for levels in lql
def quantile_loss(y_true, y_pred):
    quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(tf.cast(y_pred, "float64"), axis=1), axes=[[1], [1]]))
    error = tf.expand_dims(y_true, axis=1) - quantiles
    err1 = error * tf.expand_dims(lql - 1, 0)
    err2 = error * tf.expand_dims(lql, 0)
    loss = tf.maximum(err1, err2)
    return tf.reduce_sum(loss)
