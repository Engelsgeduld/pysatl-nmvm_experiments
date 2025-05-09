import math
from random import seed

import numpy as np
import pandas as pd
from scipy.stats import uniform, gamma


from src.estimators.semiparametric.nm_semiparametric_estimator import NMSemiParametricEstimator
from src.generators.nm_generator import NMGenerator
from src.mixtures.nm_mixture import NormalMeanMixtures
from sklearn.metrics import mean_absolute_error, mean_squared_error

seed(42)

M_SIMULATIONS = 100
N_SAMPLES_LIST = [100000]
SIGMA_0_VALUES = [1, 5, 10, 15]
SEARCH_AREA = 15.0

DISTRIBUTIONS_TO_TEST = {"Gamma": gamma(a=1, scale=1), "Uniform": uniform(0, 1)}

A_PARAM = 1 / 8
B_PARAM = 1 / 16

ALPHA_PARAM_EMPIRICAL = 0.9


def generate_nmm_sample(sigma_0, sample_len, mixing_dist_info):
    generator = NMGenerator()
    mixture = NormalMeanMixtures("canonical", sigma=sigma_0, distribution=mixing_dist_info)
    sample = generator.canonical_generate(mixture, sample_len)
    return sample


def estimate_sigma_matrix(sample, sample_len, search_area, a, b):
    log_n = math.log(sample_len)

    k = sample_len ** (1 / 2 - a - b)

    sqrt_log_n_term = math.sqrt(a * log_n / 2)
    l = (1 / search_area) * sqrt_log_n_term * k

    log_log_n = math.log(log_n)
    # eps_numerator_part = sqrt_log_n_term * k
    denominator_part = max(sample_len**b, 1.1)
    eps = ((math.sqrt(2 * a) / search_area) * (math.sqrt(log_n) / log_log_n)) / denominator_part

    estimator = NMSemiParametricEstimator(
        "sigma_estimation_eigenvalue_based",
        {"k": k, "l": l, "eps": eps, "search_area": search_area, "search_density": 100},
    )
    est = estimator.estimate(sample)

    value = est.value.real

    return value, k, l, eps


def estimate_sigma_empirical(sample, sample_len, search_area, alpha_param):
    log_n = math.log(sample_len)
    t_param_numerator = math.sqrt(alpha_param * log_n)
    t_param = t_param_numerator / (2 * search_area)

    estimator = NMSemiParametricEstimator("sigma_estimation_empirical", {"t": t_param})
    est = estimator.estimate(sample)
    value = est.value.real

    return value, t_param


df = pd.DataFrame()
matrix_conf = pd.DataFrame()
emp_conf = pd.DataFrame()

for dist_name, dist_info in DISTRIBUTIONS_TO_TEST.items():
    print(f"\n Distribution: {dist_name}")
    for n_samples in N_SAMPLES_LIST:
        print(f"\n n = {n_samples}")
        for sigma_0 in SIGMA_0_VALUES:
            print(f" sigma_0 = {sigma_0}")
            estimates_matrix_run = []
            estimates_empirical_run = []
            print(f"    Dist={dist_name}, n={n_samples}, sigma_0={sigma_0}")
            for i in range(M_SIMULATIONS):
                sample = generate_nmm_sample(sigma_0, n_samples, dist_info)
                sigma_hat_matrix, k_used, l_used, eps_used = estimate_sigma_matrix(
                    sample, n_samples, SEARCH_AREA, A_PARAM, B_PARAM
                )
                estimates_matrix_run.append(sigma_hat_matrix)
                sigma_hat_empirical, t_used = estimate_sigma_empirical(
                    sample, n_samples, SEARCH_AREA, ALPHA_PARAM_EMPIRICAL
                )
                estimates_empirical_run.append(sigma_hat_empirical)

            matrix_res_df = pd.DataFrame(
                {
                    "estimator": "matrix",
                    "distribution": dist_name,
                    "sigma": sigma_0,
                    "values": estimates_matrix_run,
                    "MAE": mean_absolute_error(np.array([sigma_0] * M_SIMULATIONS), np.array(estimates_matrix_run)),
                    "MSE": mean_squared_error(np.array([sigma_0] * M_SIMULATIONS), np.array(estimates_matrix_run)),
                }
            )
            empirical_res_df = pd.DataFrame(
                {
                    "estimator": "empirical",
                    "distribution": dist_name,
                    "sigma": sigma_0,
                    "values": estimates_empirical_run,
                    "MAE": mean_absolute_error(np.array([sigma_0] * M_SIMULATIONS), np.array(estimates_empirical_run)),
                    "MSE": mean_squared_error(np.array([sigma_0] * M_SIMULATIONS), np.array(estimates_empirical_run)),
                }
            )
            df = pd.concat([df, matrix_res_df, empirical_res_df])


df.to_csv("nm_sigma.csv")
