#!/usr/bin/env python

import pandas as pd
import numpy as np
from fcit import fcit


def nonparametric_fcit_test(X, Y, Z, data):
    """
    X and Y are names of variables.
    Z is a list of names of variables.
    data is a pandas data frame.

    Return a float corresponding to the p-value computed from FCIT.
    """

    # implement code here
    X_var = np.asmatrix(data[X].to_numpy()).transpose()
    Y_var = np.asmatrix(data[Y].to_numpy()).transpose()
    Z_var = np.asmatrix(data[Z].to_numpy())
    print(X_var.shape, Y_var.shape, Z_var.shape)
    pval = fcit.test(X_var, Y_var, Z_var)
    return pval


"""
x (n_samples, x_dim): First variable.
y (n_samples, y_dim): Second variable.
z (n_samples, z_dim): Conditioning variable. If z==None (default)
"""
def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(0)
    data = pd.read_csv("data.txt")

    print(nonparametric_fcit_test("raf", "erk", ["mek"], data))
    print(nonparametric_fcit_test("raf", "erk", ["mek", "pka", "pkc"], data))

if __name__ == "__main__":
    main()
