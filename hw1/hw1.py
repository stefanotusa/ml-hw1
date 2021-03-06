#!/usr/bin/env python

import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy.special import expit


def odds_ratio(X, Y, Z, data):
    """
    Compute the odds ratio OR(X, Y | Z).
    X, Y are names of variables
    in the data frame. Z is a list of names of variables.

    Return float OR for the odds ratio OR(X, Y | Z)
    """

    # Odds ratio = odds X = x instead of x_ref when Y = y
    #              ------------------------------------------
    #              odds X = x instead of x_ref when Y = y_ref

    #   P(Y=y | X=x, Z)             P(Y=y_ref | X=x_ref, Z)
    # -------------------   x    ---------------------------
    # P(Y=y_ref | X=x, Z)            P(Y=y | X=x_ref, Z)

    #   P(Y=1 | X=1, Z)                P(Y=0 | X=0, Z)
    # -------------------   x    ---------------------------
    #   P(Y=0 | X=1, Z)                P(Y=1 | X=0, Z)
    
    
    X_train = data[X]
    Y_train = data[Y]

    model = sm.Logit(Y_train, X_train)
    result = model.fit()

    return (result.predict(1) / (1 - result.predict(1))) * ((1 - result.predict(0)) / result.predict(0))

def compute_confidence_intervals(X, Y, Z, data, num_bootstraps=100, alpha=0.05):
    """
    Compute confidence intervals through bootstrap

    Returns tuple (q_low, q_up) for the lower and upper quantiles of the confidence interval.
    """
    
    Ql = alpha/2
    Qu = 1 - alpha/2
    estimates = []
    
    for i in range(num_bootstraps):
        # Implement your code here:
        resampled_data = data.sample(n=data.shape[0], replace=True)
        estimates.append(odds_ratio(X,Y,Z, resampled_data))

    q_low = np.quantile(estimates, Ql)
    q_up = np.quantile(estimates, Qu)
    return q_low, q_up


def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(200)
    data = pd.read_csv("data.txt") 
    print(odds_ratio("opera", "mortality", [], data), compute_confidence_intervals("opera", "mortality", [], data))
    print(odds_ratio("opera", "mortality", ["income"], data),
          compute_confidence_intervals("opera", "mortality", ["income"], data))

    print(odds_ratio("mortality", "opera", [], data), compute_confidence_intervals("opera", "mortality", [], data))
    print(odds_ratio("mortality", "opera", ["income"], data),
          compute_confidence_intervals("opera", "mortality", ["income"], data))

if __name__ == "__main__":
    main()
