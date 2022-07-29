import numpy as np
import pandas as pd
import xarray as xr

def try_out():
    n = 10
    locations = np.random.rand(n)
    scales = np.random.rand(n)
    times = pd.date_range("2021-12-03", periods=n, freq="D")

    # f: 10x2 matrix, indizierbar via Zeiten (gebe mir zu "2021-12-03" die erste location und scale)
    # und "location" oder "scale" (gebe mir die jeweilige Spalte der Matrix)
    f = xr.DataArray(data=np.array([locations, scales]).transpose()
                     , dims=["time", "features"]
                     , coords={"location": (["time"], locations), "scale": (["time"], scales),
                               "time": times}
                     # name: ([liste von dimensionen, die übrig bleiben und in "dims" stehen], DataArray)
                     )

    g = xr.DataArray(data=np.stack((locations, scales), axis=1), dims=["time", "features"],
                     coords={"time": times, "features": ["location", "scale"]})

    print(g)
    print(g.shape)
    print(g.loc[:, "location"])
    print(g.loc[times[1], :])
    print(g.sel(time=times[1]))
    print(g["time"])
    # print(f[0, :])
    # print(f["location"])
    # print(f["scale"])
    # print(f["location"][3:5])
    # print(f[3:5, "location"])  # geht nicht


def this_works_well():
    df = pd.read_csv("../data/Sweden_Zone3_Ensembles.csv")
    df["time"] = pd.to_datetime(df["time"]) - pd.to_timedelta(df["horizon"], "h")
    df = df.drop(["is_origin"], axis=1)
    df = df.pivot(index=["time", "horizon", "number"], columns=[])
    a = df.to_xarray()
    a.to_netcdf("../data/test.nc")

if __name__ == "__main__":
    a = xr.load_dataset("../data/test.nc")