import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    train_data = pd.read_csv("data/df_train.csv", index_col=0)
    # test_data = pd.read_csv("data/df_test.csv", index_col=0)

    # train_data.info()

    loc1 = train_data[train_data["location"] == 1]  # only data from location 1
    # train_data.groupby("location")["ens_mean"].mean()
    #loc1.info()
    #loc1.iloc[:,7:27].plot()
    #plt.show()
    test_data = pd.DataFrame({"loc":loc1.iloc[0],"obs":loc1.iloc[1]})
    test_data.info()
    npa =np.array(loc1.iloc[1])
    print(npa.mean())