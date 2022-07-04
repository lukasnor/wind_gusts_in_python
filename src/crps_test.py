from bqn import build_crps_loss, build_crps_loss3
import tensorflow as tf
import numpy as np
import properscoring as ps
from scipy.special import binom


def build_crps_loss(degree: int, N: int, min=0.0, max=1.0):
    # quantile levels
    lql = tf.constant(np.linspace(0, 1, N + 1), dtype="float32")
    # Bernstein polynomials to interpolate the CDF
    B = tf.constant(np.array(
        [binom(degree, j) * np.power(lql, j) * np.power(1 - lql, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")
    # shuffled quantile levels
    lql2 = tf.random.shuffle(lql)
    B2 = tf.constant(np.array(
        [binom(degree, j) * np.power(lql2, j) * np.power(1 - lql2, degree - j) for j in
         range(degree + 1)]).transpose(), dtype="float32")

    def crps(y_true, y_pred):
        # X
        q = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        # X*
        q2 = tf.transpose(tf.tensordot(B2, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
        # |X-x|
        error = (max - min) * tf.abs(q - y_true)
        # |X-X*|
        error2 = (max - min) * tf.abs(q - q2)
        return tf.reduce_mean(error, axis=1) - 0.5 * tf.reduce_mean(error2, axis=1)

    return crps


def get_quantiles(y_pred, quantile_levels):
    bernsteins = np.array(
        [binom(degree, j) * np.power(quantile_levels, j) * np.power(1 - quantile_levels, degree - j)
         for j in
         range(degree + 1)]).transpose()
    return np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]]))


degree = 12
y_pred = tf.constant([7, 6, 5, 4, 3, 2, 1, 0.1, 0.1, 0.5, 0.6, 4, 7], dtype="float", shape=(1, 13))
y_true = tf.constant([13.4], dtype="float", shape=(1, 1))

for i in range(2, 6):
    N = 10 ** i
    print(i, N)
    crps3 = build_crps_loss(degree=degree, N=N)
    print(crps3(y_true, y_pred))
    qs = np.random.random(10000)
    #qs = np.linspace(0, 1, N + 1)
    quantiles = get_quantiles(y_pred, qs)
    print(ps.crps_ensemble(y_true[0, 0], quantiles[0]))
    print()
