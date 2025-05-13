import random

import numpy as np
import pandas as pd
from scipy.stats import genhyperbolic, geninvgauss
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures

first, second = 2, 3
beta = min(first, second)
v = max(first, second)
b = np.sqrt(v**2 - beta**2)
random.seed(1)

mixture_params = {
    "mixture_form": "classical",
    "alpha": 0,
    "beta": beta,
    "gamma": 1,
    "distribution": geninvgauss(p=1, b=b, loc=0, scale=1 / b),
}
mult_t = genhyperbolic(p=1, a=v, b=beta)
mixture = NormalMeanVarianceMixtures(**mixture_params)
values = np.linspace(-1, 1, 40)
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
        "gig_pdf",
        "mixture_cdf",
        "rqmc_cdf_eps",
        "gig_cdf",
    ],
)

df["pdf_mae"] = mean_absolute_error(df["gig_pdf"], df["mixture_pdf"])
df["pdf_rmse"] = root_mean_squared_error(df["gig_pdf"], df["mixture_pdf"])
df["cdf_mae"] = mean_absolute_error(df["gig_cdf"], df["mixture_cdf"])
df["cdf_rmse"] = root_mean_squared_error(df["gig_cdf"], df["mixture_cdf"])
df["pdf_max_dif"] = np.max(np.abs(df["gig_pdf"] - df["mixture_pdf"]))
df["cdf_max_dif"] = np.max(np.abs(df["gig_cdf"] - df["mixture_cdf"]))
df["mix_moment_1"] = mixture.compute_moment(1, rqmc_params)[0]
df["gig_moment_1"] = mult_t.moment(1)
df["mix_moment_2"] = mixture.compute_moment(2, rqmc_params)[0]
df["gig_moment_2"] = mult_t.moment(2)

df.to_csv("nmv_nv.csv", index=False)
