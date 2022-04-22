from bqn import build_crps_loss
import tensorflow as tf
import numpy as np
import properscoring as ps
from scipy.special import binom

def get_quantiles(y_pred, quantile_levels):
    bernsteins = np.array(
        [binom(degree, j) * np.power(quantile_levels, j) * np.power(1 - quantile_levels, degree - j)
         for j in
         range(degree + 1)]).transpose()
    return np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]]))

degree = 12
y_pred = tf.constant([7,6,5,4,3,2,1,0.1,0.1,0.5,0.6,4,7], dtype="float", shape=(1,13))
y_true = tf.constant([13.4], dtype="float", shape=(1,1))
crps = build_crps_loss(degree=degree)
print(crps(y_true, y_pred))
qs = np.random.random(100)
print(qs)
quantiles = get_quantiles(y_pred, qs)
print(quantiles)
print(ps.crps_ensemble(y_true, quantiles))


