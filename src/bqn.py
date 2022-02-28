import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from scipy.special import binom
from keras import Model, layers
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

fontdict_title = {"fontweight": "bold", "fontsize": 18}
fontdict_axis = {"fontweight": "bold", "fontsize": 15}


# Construction of the local model
def get_local_model(name: str, n_ens_input: int, n_loc: int, n_emb: int, layer_sizes: [int], activations,
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
    model = Model(name=name, inputs=[ens_input, emb_input], outputs=output)
    return model


# Construction of model
def get_model(name: str, n_ens_input: int, layer_sizes: [int], activations,
              degree: int) -> Model:
    # Inputs
    ens_input = layers.Input(shape=n_ens_input, name="ens_input")

    # Hidden layers
    hidden1 = layers.Dense(units=layer_sizes[0], activation=activations[0], name="hidden1")(ens_input)
    hidden2 = layers.Dense(units=layer_sizes[1], activation=activations[1], name="hidden2")(hidden1)

    # Output
    output = layers.Dense(units=degree + 1, activation="softplus")(hidden2)  # smooth, non-negative and proportional

    # Model
    model = Model(name=name, inputs=ens_input, outputs=output)
    return model


# Construction of the loss function
degree = 8
loss_quantile_levels = np.arange(0.01, 1, 0.01)  # 1% to 99% quantile levels for the loss, equidistant
lql = tf.constant(loss_quantile_levels, dtype="float32")
bernsteins = np.array(
    [binom(degree, j) * np.power(loss_quantile_levels, j) * np.power(1 - loss_quantile_levels, degree - j) for j in
     range(degree + 1)]).transpose()
B = tf.constant(bernsteins, dtype="float32")


# Multi-quantile loss: mean over all quantile losses for levels in lql
def quantile_loss(y_true, y_pred):
    quantiles = tf.transpose(tf.tensordot(B, tf.cumsum(y_pred, axis=1), axes=[[1], [1]]))
    error = y_true - quantiles  # no expand dims
    err1 = error * tf.expand_dims(lql - 1, 0)
    err2 = error * tf.expand_dims(lql, 0)
    loss = tf.maximum(err1, err2)
    return tf.reduce_sum(loss, axis=1)


def get_quantiles(y_pred: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.transpose(np.tensordot(bernsteins, np.cumsum(y_pred, axis=1), axes=[[1], [1]])),
                        index=y_pred.index)


def get_rank(obs: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([obs, quantiles], axis=1).rank(axis=1).iloc[:, 0]


def generate_pit_plot(obs, quantiles) -> None:
    ranks = get_rank(obs, quantiles)
    plt.hist(ranks, density=True, bins=len(loss_quantile_levels)+1)
    plt.title("Rank Histogram", fontdict=fontdict_title)
    plt.show()


def generate_forecast_plots(y_true: pd.DataFrame, y_pred: pd.DataFrame, n=None) -> None:
    q = get_quantiles(y_pred)
    if n == None:
        n = y_true.shape[0]
    for i in range(n):
        plt.plot(xmin=0, xmax=1)
        plt.plot(q.iloc[(i, slice(None))], loss_quantile_levels, color="blue", label="forecast")
        plt.vlines(y_true.iloc[i], ymin=loss_quantile_levels.min(), ymax=loss_quantile_levels.max(),
                   label="observation", color="red")
        plt.title("Forecast " + str(i), fontdict=fontdict_title)
        plt.legend()
        plt.show()


# Import observation data
observations = pd.read_csv("../data/Sweden_Zone3_Power.csv", index_col=0)
observations.index = pd.to_datetime(observations.index, infer_datetime_format=True)
observations = observations[observations.columns.drop("horizon")]

# Import ensemble data
ensembles = pd.read_csv("../data/Sweden_Zone3_Ensembles.csv")
ensembles["time"] = pd.to_datetime(ensembles["time"], infer_datetime_format=True)
ensembles = ensembles.reset_index().pivot(index=["time", "number"], columns=[])
ensembles = ensembles.sort_index(level=[0, 1])
variables = ensembles.columns.drop(["is_origin", "horizon", "index"])
ensembles = ensembles[variables]
n_ens = len(ensembles.index.get_level_values(1).unique())

# Split train and test set
train_split = 0.8
dates = observations.index.intersection(ensembles.index.get_level_values(0).unique())
n_train = int(len(dates) * train_split)
dates_train = dates[:n_train]
dates_test = dates[n_train:]
ens_train = ensembles.loc[(dates_train, slice(None))]
ens_test = ensembles.loc[(dates_test, slice(None))]
obs_train = observations.loc[dates_train]
obs_test = observations.loc[dates_test]

# Scale ensembles
ens_scaler = MinMaxScaler()
ens_scaler.fit(ens_train)
sc_ens_train = pd.DataFrame(data=ens_scaler.transform(ens_train), index=ens_train.index, columns=ens_train.columns)
sc_ens_test = pd.DataFrame(data=ens_scaler.transform(ens_test), index=ens_test.index, columns=ens_test.columns)

# Scale observations
obs_scaler = MinMaxScaler()  # MinMaxScaler more suitable for power data. But even better when not aggregated
obs_scaler.fit(obs_train)
sc_obs_train = pd.DataFrame(data=obs_scaler.transform(obs_train), index=obs_train.index, columns=obs_train.columns)
sc_obs_test = pd.DataFrame(data=obs_scaler.transform(obs_test), index=obs_test.index, columns=obs_test.columns)

# Compile model
model = get_model(name="first_try", n_ens_input=n_ens, layer_sizes=[48, 24],
                  activations=["selu", "selu"], degree=degree)
model.compile(optimizer="adam", loss=quantile_loss)

sc_ens_train = sc_ens_train["speed"].reset_index().pivot(index="time", columns="number")
history = model.fit(y=sc_obs_train,
                    x=sc_ens_train,
                    batch_size=20,
                    epochs=500,
                    verbose=1,
                    validation_freq=1,
                    validation_split=0.10,
                    callbacks=[EarlyStopping(patience=20, restore_best_weights=True)],
                    use_multiprocessing=True
                    )
sc_ens_test = sc_ens_test["speed"].reset_index().pivot(index="time", columns="number")
test = pd.DataFrame(model.predict(sc_ens_test), index=sc_ens_test.index)
generate_forecast_plots(sc_obs_test, test, 10)
generate_pit_plot(sc_obs_test, get_quantiles(test))
