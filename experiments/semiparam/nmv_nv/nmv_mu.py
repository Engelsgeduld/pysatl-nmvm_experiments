from random import seed

import numpy as np
from scipy.stats import expon, pareto
import pandas as pd

from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures
from src.generators.nmv_generator import NMVGenerator
from src.estimators.semiparametric.nmv_semiparametric_estimator import NMVSemiParametricEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error

seed(42)


def generate_nmvm_samples(real_mu: float, n: int, mixing_dist) -> np.ndarray:
    mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=real_mu, distribution=mixing_dist)
    generator = NMVGenerator()
    sample = generator.canonical_generate(mixture, n)
    return sample


REAL_MU = [0, 1, 2, 5]
SAMPLE_SIZES = 100000
NUM_SIMULATIONS = 100

distributions = [(expon(1), "expon"), (pareto(2), "pareto")]

ESTIMATOR_PARAMS_MU = {
    "m": 10,
    "tolerance": 1e-5,
}


estimator = NMVSemiParametricEstimator("mu_estimation", ESTIMATOR_PARAMS_MU)

df = pd.DataFrame()

for mu in REAL_MU:
    for distribution, distrib_name in distributions:
        estimates_mu_list = []
        for i in range(NUM_SIMULATIONS):
            sample = generate_nmvm_samples(mu, SAMPLE_SIZES, distribution)

            estimate_result = estimator.estimate(sample)
            estimates_mu_list.append(estimate_result.value)

        res_df = pd.DataFrame(
            {
                "mu": mu,
                "distribution": distrib_name,
                "values": estimates_mu_list,
                "MAE": mean_absolute_error(np.array([mu] * NUM_SIMULATIONS), np.array(estimates_mu_list)),
                "MSE": mean_squared_error(np.array([mu] * NUM_SIMULATIONS), np.array(estimates_mu_list)),
            }
        )
        df = pd.concat([df, res_df])

df.to_csv("nmv_nv_mu.csv")
