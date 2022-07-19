import tensorflow as tf
import numpy as np


def build_hen_crps(bin_edges: np.ndarray):
    b = tf.constant(bin_edges, dtype="float")
    N = len(bin_edges) - 1

    def crps(y_true, y_pred):
        y_tilde = tf.minimum(tf.expand_dims(b[1:], axis=0),
                             tf.maximum(tf.expand_dims(b[:-1], axis=0),
                                        tf.expand_dims(y_true, axis=1)))
        L = tf.cumsum(y_pred, axis=1, exclusive=True)
        d = b[1:] - b[:-1]
        first = tf.pow(L, 2) * (y_tilde - b[:-1])
        second = tf.pow(1 - L, 2) * (b[1:] - y_tilde)
        third = y_pred / d * tf.pow(y_tilde - b[:-1], 2)
        forth = y_pred * d * (L - 1 + y_pred / 3)
        fifth = tf.abs(y_true - tf.minimum(tf.maximum(y_true, tf.expand_dims(b[0], axis=0)),
                                           tf.expand_dims(b[N], axis=0)))
        return tf.reduce_mean(tf.reduce_sum(first + second + third + forth, axis=1) + fifth)

    return crps


if __name__ == "__main__":
    bin_edges = np.array(range(11), dtype="float")
    b = tf.constant(bin_edges, dtype="float")
    N = len(bin_edges) - 1
    y_true = tf.constant([0.15, 9.75], shape=(2,))
    y_pred = tf.constant([[0.0, 0.0, 0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.4],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.15, 0.2, 0.2, 0.35]],
                         shape=(2, N))
    y_tilde = tf.minimum(tf.expand_dims(b[1:], axis=0),
                         tf.maximum(tf.expand_dims(b[:-1], axis=0),
                                    tf.expand_dims(y_true, axis=1)))
    L = tf.cumsum(y_pred, axis=1, exclusive=True)
    U = 1 - tf.cumsum(y_pred, axis=1, reverse=True, exclusive=True) # U - L = y_pred
    d = b[1:] - b[:-1]
    first = tf.pow(L, 2) * (y_tilde - b[:-1])
    second = tf.pow(1 - L, 2) * (b[1:] - y_tilde)
    third = y_pred / d * tf.pow(y_tilde - b[:-1], 2)
    forth = y_pred * d * (L - 1 + y_pred / 3)
    fifth = tf.abs(y_true - tf.minimum(tf.maximum(y_true, tf.expand_dims(b[0], axis=0)), tf.expand_dims(b[N], axis=0)))
    result = tf.reduce_sum(first + second + third + forth, axis=1)+fifth