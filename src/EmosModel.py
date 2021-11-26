import time
from abc import ABC

import numpy as np
from pywatts.core.base import BaseEstimator
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri

# Import the 'scoringRules' library from R, since yet no worthy equivalent exists for Python
from scipy.optimize import minimize

r = robjects.r
r['source']('scoringRules.R')
crps_tlogis_R = robjects.globalenv['cprs_tlogis']
gradcrps_tlogis_R = robjects.globalenv['gradcrps_tlogis']
numpy2ri.activate()


def crps_tlogis(y, location=0, scale=1, lower=-np.inf, upper=np.inf):
    if lower == -np.inf or upper == np.inf:
        print("Attention: I dont know if the conversion of numpy to R works as expected.")
    result = np.array(
        crps_tlogis_R(y, location, scale, lower, upper))  # This should give a numpy array of shape (len(y))
    print("Attention: I dont know if the conversion from R to numpy works as expected.")
    return result


def gradcrps_tlogis(y, location=0, scale=1, lower=-np.inf, upper=np.inf):
    if lower == -np.inf or upper == np.inf:
        print("Attention: I dont know if the conversion of numpy to R works as expected.")
    result = np.array(
        gradcrps_tlogis_R(y, location, scale, lower, upper))  # This should give a numpy array of unknown shape
    print("Attention: I dont know if the conversion from R to numpy works as expected.")
    return result


class EmosModel(BaseEstimator, ABC):

    # Inputs:
    # name: Name of the model
    # train: Training data
    # n_ens: Number of ensembles

    # TODO: All variables should be given in the self.fit(**kwargs) method. Here only the declaration.
    # Maybe global variables are faster, but it is what it is for now
    def __init__(self, name, n_ens: int = 20, model_type: str = "normal", par_start=[0, 0, 0, 1], t_sd=0, **kwargs):
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
        super.__init__(name)

    def fn_loc(self, a, b):
        return min(-self.t_max, np.minimum(self.t_max, a + np.exp(b) * self.ens_mean))

    def fn_scale(self, c, d):
        return max(self.t_min, np.minimum(self.t_max, np.exp(c + d * np.log(self.ens_sd))))

    def fn_sr(self, location, scale):
        return crps_tlogis(y=self.obs, location=location, scale=scale, lower=0)

    def fn_grad(self):
        return

    def wrapper(self, par_emos):
        #### Calculation ####
        # Calculate location and scale parameters
        loc_emos = self.fn_loc(  # ens_mean=self.ens_mean,
            a=par_emos[1],
            b=par_emos[2])
        scale_emos = self.fn_scale(  # ens_sd=self.ens_sd,
            c=par_emos[3],
            d=par_emos[4])

        # Calculate mean scores of training data
        res = np.mean(self.fn_sr(y=self.obs,
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
        s_grad = self.fn_grad(y=self.obs, location=loc_emos, scale=scale_emos)

        # Derivatives w.r.t. a and b
        print("EmosModel.grad: I dont know yet, what fn_grad returns.")
        raise Exception()
        result[0] = None
        result[1] = None

        # Derivatives w.r.t. c and d
        result[2] = None
        result[3] = None

        return result

    # Input
    def fit(self, **kwargs):

        names = ["obs", "ens_mean", "ens_sd"]

        # Check for complete input
        for name in names:
            if name not in kwargs.keys():
                print("Emos-fit:", name, "not in parameters.")

        self.obs = np.array(kwargs["obs"])
        self.ens_mean = np.array(kwargs["ens_mean"])
        self.ens_sd = np.array(kwargs["ens_sd"])

        # aggregate
        # train = pd.DataFrame({"obs":obs,"ens_mean":ens_mean,"ens_sd":ens_sd})

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
        estimation = minimize(self.wrapper, self.par_start, method='BFGS', jac=self.grad,
                              options={'disp': True})
        stop_time = time.time()

        return [estimation.x, len(self.obs), str(stop_time - start_time)+" seconds"]

    def load(cls, load_information) -> BaseEstimator:
        return EmosModel(load_information)
