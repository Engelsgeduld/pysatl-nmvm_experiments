from random import seed

import numpy as np
import pandas as pd
from scipy.stats import expon, pareto
from src.mixtures.nmv_mixture import NormalMeanVarianceMixtures
from src.generators.nmv_generator import NMVGenerator
from src.estimators.semiparametric.nmv_semiparametric_estimator import NMVSemiParametricEstimator
from sklearn.metrics import mean_absolute_error

seed(42)

df = pd.DataFrame()

for mu in [0, 1]:
    for distribution, dist_name in [(expon(1), "expon"), (pareto(2), "pareto")]:
        mixture = NormalMeanVarianceMixtures("canonical", alpha=0, mu=mu, distribution=distribution)
        generator = NMVGenerator()
        sample = generator.canonical_generate(mixture, 10000)

        x_values = np.linspace(0.1, 3, 100000)
        estimator_given_mu = NMVSemiParametricEstimator(
            "g_estimation_given_mu", {"x_data": x_values, "u_value": 7.6, "v_value": 0.9, "mu": mu, "grid_size": 200}
        )
        estimator_post_widder = NMVSemiParametricEstimator(
            "g_estimation_post_widder", {"x_data": x_values, "sigma": 1, "n": 2, "mu": mu}
        )

        result_given_mu = estimator_given_mu.estimate(sample)
        result_post_widder = estimator_post_widder.estimate(sample)

        result_post_widder_float = [float(val) for val in result_post_widder.list_value]

        res_post_mellin_df = pd.DataFrame(
            {
                "x": x_values,
                "mu": mu,
                "method": "Mellin",
                "distribution": dist_name,
                "real_g": distribution.pdf(x_values),
                "estimated_g": result_given_mu.list_value,
                "MAE": mean_absolute_error(distribution.pdf(x_values), result_given_mu.list_value),
            }
        )
        res_post_mellin_df["max_dif"] = np.max(np.abs(res_post_mellin_df["real_g"] - res_post_mellin_df["estimated_g"]))

        res_post_widder_df = pd.DataFrame(
            {
                "x": x_values,
                "mu": mu,
                "method": "Post Widder",
                "distribution": dist_name,
                "real_g": distribution.pdf(x_values),
                "estimated_g": result_post_widder_float,
                "MAE": mean_absolute_error(distribution.pdf(x_values), result_post_widder_float),
            }
        )
        res_post_widder_df["max_dif"] = np.max(np.abs(res_post_widder_df["real_g"] - res_post_widder_df["estimated_g"]))

        df = pd.concat([df, res_post_mellin_df, res_post_widder_df])


df.to_csv("nmv_nv_g.csv")
