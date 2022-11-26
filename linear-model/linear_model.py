import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm


# Constant for later use.
COLOURS = ["skyblue", "black", "blue"]
CI = 0.9

# Create some data.
X = np.random.normal(0.0, 1.0, 100)
y = 1 + 2 * X + np.random.normal(0.0, 1.0, len(X))

df = pd.DataFrame({"X": X, "y": y})
df.to_csv("df.csv", index=False)

with pm.Model() as lm:
    # Priors.
    alpha = pm.Normal("alpha", 0.0, 0.5)
    beta = pm.Normal("beta", 0.0, 1.0)
    sigma = pm.Exponential("sigma", 1.0)
    # Likelihood
    mu = alpha + beta * df["X"].to_numpy()
    y = pm.Normal("y", mu, sigma, observed=df["y"].to_numpy())
    # Check the prior.
    idata_lm = pm.sample_prior_predictive()

# Checking the prior model structure and predictions.
pm.model_to_graphviz(lm).render("lm_DAG", format="png", cleanup=True)

az.plot_ppc(
    idata_lm,
    kind="cumulative",
    colors=COLOURS,
    group="prior",
    observed=False,
)
plt.savefig("lm_prior_check.png")

with lm:
    idata_lm.extend(pm.sample(1000, chains=5))
    idata_lm.extend(pm.sample_posterior_predictive(idata_lm))

az.plot_ppc(idata_lm, kind="cumulative", colors=COLOURS)
plt.savefig("lm_posterior_check.png")

az.summary(idata_lm, hdi_prob=CI, stat_focus="median").to_csv("lm_summary.csv")

az.plot_trace(idata_lm, compact=True)
plt.savefig("traceplot.png")

az.plot_posterior(idata_lm, point_estimate="median", hdi_prob=CI)
plt.savefig("lm_posteriors.png")

az.plot_dist_comparison(idata_lm)
plt.savefig("distribution_comparisons.png")

az.plot_pair(idata_lm, var_names=["alpha", "beta"])
plt.savefig("pairplot_alpha_beta.png")
