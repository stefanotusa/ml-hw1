#!/usr/bin/env python

# imports that are potentially quite useful
import numpy as np
import pandas as pd
from scipy.linalg import expm

from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from random import randint
import random

# comment these lines if you want to visualize
# your final DAG using something other than Ananke
#from ananke.graphs import DAG


def directed_cycle_score(A):
    """
    Compute a score for the number of directed cycles in a graph
    given its binary adjacency matrix A. Returns a float that is
    non-negative and is zero if and only if the graph is a DAG.
    """

    # Implement your cycle score given Problem 4 Part 2
    temp_matrix = np.zeros(A.shape)
    alpha = 0.05
    k = 0
    summation_term = 999999
    num_terms = A.shape[0]
    # while change < 0.05:
    for i in range(num_terms):
        summation_term = (1 / np.math.factorial(k)) * expm(A)
        temp_matrix += summation_term

    cycle_score = np.trace(temp_matrix) - (A.shape[0] * num_terms)
    return cycle_score

def bic_score(A, data, idx_to_var_map):
    """
    Compute the BIC score for a DAG given by its adjacency matrix A
    and data as a pandas data frame. idx_to_var_map is a dictionary
    mapping row/column indices in the adjacency matrix to variable names
    in the data frame
    """

    bic = 0
    num_vars = len(A)
    # Implement this function given Problem 4 Part 3

    # iterate over all variables
    for i in range(num_vars):

        # fit OLS model for each variable given its parents
        # As an e.g. of OLS usage: if I wanted to fit an OLS model
        # for X as a function of an intercept, Y and Z
        # I would use model = ols(formula="X ~ 1 + Y + Z", data=data).fit()
        # The BIC can then be obtained as model.bic

        # Build formula for regression on a variable given its parents and an intercept term.
        v_i = idx_to_var_map[i]
        formula = v_i + " ~ 1"

        # Find parents of a given variable
        parents_of_v_i = []
        for j in range(num_vars):
            if A[j][i] == 1:
                parents_of_v_i.append(idx_to_var_map[j])
        
        # Add them to regression
        for parent in parents_of_v_i:
            formula += " + " + parent 

        model = ols(formula=formula, data=data).fit()
        bic += model.bic

    return bic

def causal_discovery(data, num_steps=100, cycle_score_tolerance=1e-9):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.
    Since the output of the cycle score function is a float, comparison to
    zero is a little tricky. We use a really small tolerance to say the number
    is close enough to zero. That is, for x < cycle_score_tolerance x is close
    enough to 0.
    """
    idx_to_var_map = {i: var_name for i, var_name in enumerate(data.columns)}
    num_vars = len(data.columns)

    # initialize an empty graph
    A_opt = np.zeros((num_vars, num_vars), dtype=int)
    # besides the adjacency matrix keep a set of edges present
    # in the graph making for easy delete/reverse moves. each entry in the
    # set is a tuple of integers (i, j) corresponding to indices
    # for the end points of a directed edge Vi-> Vj
    edges = set([])

    # get initial BIC score for empty graph and set it to the current optimal
    bic_opt = bic_score(A_opt, data, idx_to_var_map)

    for step in range(num_steps):

        # See details in Algorithm 1 of the hw3 handout
        # for what to do in this for loop
        pass

    return A_opt, edges, idx_to_var_map


def test_causal_discovery_function():
    ################################################
    # Tests for your causal_discovery function
    ################################################
    np.random.seed(1000)
    random.seed(0)
    data = pd.read_csv("data.txt")
    A_opt, edges, idx_to_var_map = causal_discovery(data, num_steps=100)

    # comment these lines if visualizing the DAG using something other
    # than Ananke. Make sure to supply alternative visualization
    # code though!
    vertices = idx_to_var_map.values()
    edges = [(idx_to_var_map[i], idx_to_var_map[j]) for i, j in edges]
    G = DAG(vertices, edges)
    # the DAG will be stored in a PDF final_DAG.gv.pdf
    G.draw().render("final_DAG.gv", view=False)


def test_directed_cycle_score_function():
    ################################################
    # Tests for your directed_cycle_score function
    ################################################

    # Treating X, Y, Z as indices 0, 1, 2 in the adjacency matrix
    # X->Y<-Z, Z->X
    A1 = np.array([[0, 1, 0],
                   [0, 0, 0],
                   [1, 1, 0]])

    # X->Y->Z, Z->X
    A2 = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])

    print(directed_cycle_score(A1))
    print(directed_cycle_score(A2))

def test_bic_score_function():
    ################################################
    # Tests for your bic_score function
    ################################################
    data = pd.read_csv("bic_test_data.txt")
    idx_to_var_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    # fit model for G1: A->B->C->D, B->D and get BIC
    # you can also use this as additional tests for your cycle score function
    A1 = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A1, data, idx_to_var_map), directed_cycle_score(A1))


    # fit model for G2: A<-B->C->D, B->D and get BIC
    A2 = np.array([[0, 0, 0, 0],
                   [1, 0, 1, 1],
                   [0, 0, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A2, data, idx_to_var_map), directed_cycle_score(A2))

    # fit model for G3: A->B<-C->D, B->D and get BIC
    A3 = np.array([[0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 1],
                   [0, 0, 0, 0]])
    print(bic_score(A3, data, idx_to_var_map), directed_cycle_score(A3))

    # fit model for G4: A<-B->C<-D, B->D and get BIC
    A4 = np.array([[0, 0, 0, 0],
                   [1, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 1, 0]])
    print(bic_score(A4, data, idx_to_var_map), directed_cycle_score(A4))



def main():
    # This function must be left unaltered at submission (e.g. you can fiddle with this function for debugging, but return it to its original state before you submit)
    #test_causal_discovery_function()
    #test_directed_cycle_score_function()
    test_bic_score_function()

if __name__ == "__main__":
    main()
