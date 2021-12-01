import time
from typing import Dict
import numpy as np
import xarray as xr
from pywatts.core.base import BaseEstimator
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# Import the 'scoringRules' library from R, since yet no worthy equivalent exists for Python
from scipy.optimize import minimize

r = robjects.r
r['source']('src/scoringRules.R')
crps_tlogis_R = robjects.globalenv['crps_tlogis']
gradcrps_tlogis_R = robjects.globalenv['gradcrps_tlogis']
numpy2ri.activate()


def crps_tlogis(y, location=0, scale=1, lower=-np.inf, upper=np.inf):
    return np.array(
        crps_tlogis_R(y, location, scale, lower, upper))  # This should give a numpy array of shape (len(y))


def gradcrps_tlogis(y, location=0, scale=1, lower=-np.inf, upper=np.inf):
    return np.array(
        gradcrps_tlogis_R(y, location, scale, lower, upper))  # This should give a numpy array of unknown shape


class EmosModel(BaseEstimator):

    # TODO: All variables should be given in the self.fit(**kwargs) method. Here only the declaration. Or should they?
    # Maybe global variables are faster, but it is what it is for now.
    def __init__(self, name: str, n_ens: int = 20, model_type: str = "tlogis", par_start=[0, 0, 0, 1], t_sd=0):
        self.n_ens = n_ens
        self.obs = None
        self.ens_mean = None
        self.ens_sd = None

        self.model_type = model_type
        self.par_start = par_start
        self.t_sd = t_sd
        # Threshold for Nelder-Mead improvement
        self.t_nelder = 1e-3

        # Minimum resp. maximum threshold for location and scale
        self.t_max = 1e+3
        self.t_min = 1e-3

        # Threshold for (almost) zero observations
        self.t_0 = 1e-2
        super().__init__(name=name)

    def get_params(self) -> Dict[str, object]:
        return {"name": self.name,
                "n_ens": self.n_ens,
                "model_type": self.model_type,
                "par_start": self.par_start,
                "t_sd": self.t_sd}

    def set_params(self, **kwargs):
        self.n_ens = kwargs["n_ens"]
        self.model_type = kwargs["model_type"]
        self.par_start = kwargs["par_start"]
        self.t_sd = kwargs["t_sd"]

    def fn_loc(self, a, b):
        return np.maximum(-self.t_max, np.minimum(self.t_max, a + np.exp(b) * self.ens_mean))

    def fn_scale(self, c, d):
        return np.maximum(self.t_min, np.minimum(self.t_max, np.exp(c + d * np.log(self.ens_sd))))

    def fn_sr(self, location, scale):
        return crps_tlogis(y=self.obs, location=location, scale=scale, lower=0)

    def fn_grad(self, location, scale):
        return gradcrps_tlogis(y=self.obs, location=location, scale=scale, lower=0)

    def wrapper(self, par_emos):
        #### Calculation ####
        # Calculate location and scale parameters
        loc_emos = self.fn_loc(  # ens_mean=self.ens_mean,
            a=par_emos[0],
            b=par_emos[1])
        scale_emos = self.fn_scale(  # ens_sd=self.ens_sd,
            c=par_emos[2],
            d=par_emos[3])

        # Calculate mean scores of training data
        res = np.mean(self.fn_sr(  # y=self.obs,
            location=loc_emos,
            scale=scale_emos))

        # TODO: Beautify the code later
        # Output
        # var_check(y=res,
        #         check_type="finite",
        #          name_function="wrapper in emos_est")
        # var_check(y=res,
        #          name_function="wrapper in emos_est")
        return res

    def grad(self, par_emos):

        result = [0, 1, 0, 1]
        # Calculate location and scale parameters for each ensemble
        loc_emos = self.fn_loc(a=par_emos[0], b=par_emos[1])
        scale_emos = self.fn_scale(c=par_emos[2], d=par_emos[3])

        # Calculate gradient of CRPS
        s_grad = self.fn_grad(  # y=self.obs,
            location=loc_emos, scale=scale_emos)

        # Derivatives w.r.t. a and b
        result[0] = np.nanmean(s_grad[:, 0])
        result[1] = np.nanmean(s_grad[:, 0] * np.exp(par_emos[1]) * self.ens_mean)

        # Derivatives w.r.t. c and d
        result[2] = np.nanmean(s_grad[:, 1] * scale_emos)
        result[3] = np.nanmean(s_grad[:, 1] * scale_emos * np.log(self.ens_sd))

        return result

    def fit(self, **kwargs):

        names = ["obs", "ens_mean", "ens_sd"]

        # Check for complete input
        for name in names:
            if name not in kwargs.keys():
                print("Emos-fit:", name, "not in parameters.")
                raise Exception()

        self.obs = np.array(kwargs["obs"])
        self.ens_mean = np.array(kwargs["ens_mean"])
        self.ens_sd = np.array(kwargs["ens_sd"])

        #### Data preparation ####
        # Cut ensemble standard deviations to t_sd
        if not self.t_sd == 0:
            self.ens_sd[self.ens_sd > self.t_sd] = self.t_sd

        # Set (almost) zero-observations to t_0
        # In contrast to the R version, no new name "obs_cut" is introduced
        self.obs[self.obs < self.t_0] = self.t_0
        # obs_cut = obs

        #### Estimation ####
        # Set initial values (a = c = 0, b/exp(b) = d = 1, i.e. ensemble mean and standard deviation as parameters)

        start_time = time.time()
        estimation = minimize(self.wrapper, np.array(self.par_start), method='BFGS', jac=self.grad,
                              options={'disp': True})
        stop_time = time.time()

        print([estimation.x, len(self.obs), str(stop_time - start_time) + " seconds"])
        return [estimation.x, len(self.obs), str(stop_time - start_time) + " seconds"]

    # Stub
    def transform(self, **kwargs: Dict[str, xr.DataArray]):
        print("EmosModel transform function called with parameters:")
        print(kwargs)
        return xr.DataArray(data=[0, 1], dims=["x"])
