import random

import numpy as np
import pandas as pd
from scipy.stats import halfnorm, skewnorm
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.mixtures.nm_mixture import NormalMeanMixtures

a, loc, scale = 5, 0, 1
delta = a / np.sqrt(1 + a**2)
random.seed(1)

mixture_params = {
    "mixture_form": "classical",
    "alpha": loc,
    "beta": scale * delta,
    "gamma": scale * np.sqrt(1 - delta**2),
    "distribution": halfnorm(),
}
mult_t = skewnorm(a=a, loc=loc, scale=scale)
mixture = NormalMeanMixtures(**mixture_params)
values = np.linspace(-2, 2, 40)
rqmc_params = {"error_tolerance": 0.001, "i_max": 300}

result = np.array(
    [
        values,
        np.vectorize(mixture.compute_pdf)(values, rqmc_params)[0].T,
        mult_t.pdf(values).T,
        np.vectorize(mixture.compute_cdf)(values, rqmc_params)[0].T,
        mult_t.cdf(values).T,
    ]
)

df = pd.DataFrame(
    result.T,
    columns=[
        "x",
        "mixture_pdf",
        "skew_pdf",
        "mixture_cdf",
        "skew_cdf",
    ],
)

df["pdf_mae"] = mean_absolute_error(df["skew_pdf"], df["mixture_pdf"])
df["pdf_rmse"] = root_mean_squared_error(df["skew_pdf"], df["mixture_pdf"])
df["cdf_mae"] = mean_absolute_error(df["skew_cdf"], df["mixture_cdf"])
df["cdf_rmse"] = root_mean_squared_error(df["skew_cdf"], df["mixture_cdf"])
df["pdf_max_dif"] = np.max(np.abs(df["skew_pdf"] - df["mixture_pdf"]))
df["cdf_max_dif"] = np.max(np.abs(df["skew_cdf"] - df["mixture_cdf"]))
df["mix_moment_1"] = mixture.compute_moment(1, rqmc_params)[0]
df["skew_moment_1"] = mult_t.moment(1)
df["mix_moment_2"] = mixture.compute_moment(2, rqmc_params)[0]
df["skew_moment_2"] = mult_t.moment(2)

df.to_csv("nmm.csv", index=False)
