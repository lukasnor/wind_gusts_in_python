import pandas as pd
import tensorflow as tf
from keras import layers, Model
from sklearn.preprocessing import StandardScaler

observations: pd.DataFrame = pd.read_csv("../data/Offshore_Observations.csv", index_col=0)  # time is index
observations.index = pd.to_datetime(observations.index)  # convert Index to DateTimeIndex
pred_vars = observations.keys().drop(
    ["horizon", "is_origin"])  # Index(['u10', 'v10', 'd2m', 't2m', 'msl', 'sp', 'speed'])
observations = observations[pred_vars]  # leave out "horizon" and "is_origin" from observations
observations = observations.sort_index(level=0)

ensembles: pd.DataFrame = pd.read_csv("../data/Offshore_Ensembles.csv")
ensembles["time"] = pd.to_datetime(ensembles["time"], infer_datetime_format=True)  # convert time column to datetime
ensembles = ensembles.pivot(index=["horizon", "time", "number"], columns=[])  # create multiindex
ensembles = ensembles[pred_vars]  # reduce columns to necessary ones
ensembles = ensembles.sort_index(
    level=[0, 1, 2])  # sort by horizon first (irrelevant), then by date (relevant for iloc!)

horizon = 18
ensembles = ensembles.loc[horizon]  # select horizon from data
print(ensembles.index.get_level_values(0).unique().__len__())
observations = observations.loc[
    ensembles.index.get_level_values(0).unique()]  # only use the observations corresponding to the forecasts

n_obs = len(observations)  # 577
n_ens = ensembles.index.levshape[1]
split = 0.75
n_train_split = int(split * n_obs)  # number of dates
train = pd.DataFrame(ensembles.iloc[:n_train_split * n_ens])  # split test and train data
test = pd.DataFrame(ensembles.iloc[n_train_split * n_ens:])

scaler = StandardScaler()
scaler.fit(train)
train_norm = pd.DataFrame(data=scaler.transform(train), index=train.index, columns=train.columns)
test_norm = pd.DataFrame(data=scaler.transform(test), index=test.index, columns=test.columns)
observations_norm = pd.DataFrame(data=scaler.transform(observations), index=observations.index,
                                 columns=observations.columns)

train_speed = train_norm.reset_index().pivot(index="time", columns="number", values="speed")
test_speed = test_norm.reset_index().pivot(index="time", columns="number", values="speed")


input = layers.Input(name="input", shape=n_ens)
mean_model = Model(name="mean_model", inputs=input,
                   outputs=layers.Lambda(name="mean_layer", function=(lambda ens: tf.reduce_mean(ens, axis=1)), output_shape=1)(input))
mean_model.trainable = False
mean_model.compile(optimizer="adam", loss='mean_absolute_error')
print("Evaluation of mean_model")
mean_model.evaluate(x=test_speed, y=observations["speed"].iloc[n_train_split:])  # 6.9108
print()