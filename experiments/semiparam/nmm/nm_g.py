from random import seed

from src.estimators.semiparametric.nm_semiparametric_estimator import NMSemiParametricEstimator
from src.generators.nm_generator import NMGenerator
from src.mixtures.nm_mixture import NormalMeanMixtures
import numpy as np
from scipy.stats import gamma, uniform
import pandas as pd
from sklearn.metrics import mean_absolute_error

seed(42)


def generate_nmm_sample(sigma_0, sample_len, mixing_dist):
    generator = NMGenerator()
    mixture = NormalMeanMixtures("canonical", sigma=sigma_0, distribution=mixing_dist)
    sample = generator.canonical_generate(mixture, sample_len)
    return sample


df = pd.DataFrame()
n_samples = 1000000
sigma_0 = 1
x = np.linspace(0, 20, 100000)

for mixing_dist, name in [(gamma(a=1, scale=1), "gamma"), (uniform(0, 1), "uniform")]:
    sample = generate_nmm_sample(sigma_0, n_samples, mixing_dist)
    estimator = NMSemiParametricEstimator(
        "g_estimation_convolution", {"x_data": x, "sigma": sigma_0, "bohman_n": 10000, "bohman_delta": 0.0001}
    )
    result = estimator.estimate(sample)
    res_df = pd.DataFrame(
        {
            "x": x,
            "mix": name,
            "real_g": mixing_dist.pdf(x),
            "estimated_g": result.list_value,
            "MAE": mean_absolute_error(mixing_dist.pdf(x), result.list_value),
        }
    )
    res_df["max_dif"] = np.max(np.abs(res_df["real_g"] - res_df["estimated_g"]))
    df = pd.concat([df, res_df])

df.to_csv("nm_g.csv")
