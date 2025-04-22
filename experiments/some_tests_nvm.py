import random

import numpy as np
import pandas as pd
from scipy.stats import invgamma, t
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.mixtures.nv_mixture import NormalVarianceMixtures

loc, scale, v = 0, 1, 5
random.seed(1)

mixture_params = {"mixture_form": "classical", "alpha": loc, "gamma": 1, "distribution": invgamma(v / 2, scale=v / 2)}
mult_t = t(df=v, loc=loc)
mixture = NormalVarianceMixtures(**mixture_params)
interval = mult_t.interval(0.8)
values = np.linspace(-2, 2, 40)
rqmc_params = {"error_tolerance": 0.001, "i_max": 300}

pdf_result = np.vectorize(mixture.compute_pdf)(values, rqmc_params)
cdf_result = np.vectorize(mixture.compute_cdf)(values, rqmc_params)

result = np.array(
    [
        values,
        pdf_result[0].T,
        pdf_result[1].T,
        mult_t.pdf(values).T,
        cdf_result[0].T,
        cdf_result[1].T,
        mult_t.cdf(values).T,
    ]
)

df = pd.DataFrame(
    result.T,
    columns=[
        "x",
        "mixture_pdf",
        "rqmc_pdf_eps",
        "t_pdf",
        "mixture_cdf",
        "rqmc_cdf_eps",
        "t_cdf",
    ],
)

df["pdf_mae"] = mean_absolute_error(df["t_pdf"], df["mixture_pdf"])
df["pdf_rmse"] = root_mean_squared_error(df["t_pdf"], df["mixture_pdf"])
df["cdf_mae"] = mean_absolute_error(df["t_cdf"], df["mixture_cdf"])
df["cdf_rmse"] = root_mean_squared_error(df["t_cdf"], df["mixture_cdf"])
df["pdf_max_dif"] = np.max(np.abs(df["t_pdf"] - df["mixture_pdf"]))
df["cdf_max_dif"] = np.max(np.abs(df["t_cdf"] - df["mixture_cdf"]))
df["mix_moment_1"] = mixture.compute_moment(1, rqmc_params)[0]
df["t_moment_1"] = mult_t.moment(1)
df["mix_moment_2"] = mixture.compute_moment(2, rqmc_params)[0]
df["t_moment_2"] = mult_t.moment(2)

df.to_csv("nvm.csv", index=False)
