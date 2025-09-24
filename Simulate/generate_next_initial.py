import numpy as np
import pandas as pd

def simulate_complete_linear(N=1000, seed=0):
    rng = np.random.default_rng(seed)

    # Parameters for predictors
    mu = np.array([25.0, 27.0, 29.0])     # means
    Sigma = np.array([[12.0,  4.0,  2.0],  # covariance matrix (positive-definite)
                      [ 4.0, 10.0,  3.0],
                      [ 2.0,  3.0,  8.0]])

    # Draw predictors
    X = rng.multivariate_normal(mu, Sigma, size=N)
    x1, x2, x3 = X.T

    # True regression coefficients
    beta = np.array([90.0, 3.0, 7.0, -2.0])  # [intercept, b1, b2, b3]
    sigma = 15.0                             # noise SD

    # Generate response
    eps = rng.normal(0.0, sigma, size=N)
    y = beta[0] + beta[1]*x1 + beta[2]*x2 + beta[3]*x3 + eps

    # Put in DataFrame
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    truth = {"beta": beta, "sigma": sigma, "mu": mu, "Sigma": Sigma}
    return df, truth

# Example use:
df, truth = simulate_complete_linear(N=1000, seed=42)
print(df.head())
print(truth)
df.to_csv("Data/Inputs/next_initial_simulated_data.csv", index=False)
