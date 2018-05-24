# -*- coding: utf-8 -*-
"""
Created on Thu May  3 12:52:55 2018

@author: nicol
"""

import numpy as np
import sklearn.gaussian_process as gp
import numpy.linalg as np
from scipy.spatial import distance
import heapq
from random import randint
import random
from random import *
import sys
import sklearn
from sklearn.svm import SVR
import numpy as np
from meoadubqp import *
from ZDT1_problem import *
from evo_op import *
from Aggreg import *

from scipy.stats import norm
from scipy.optimize import minimize

def expected_improvement(x, gaussian_process, evaluated_loss, n_params ,greater_is_better=False):
    """ expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """
    
    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement



def moead_EI(gaussian_processes , evaluated_losses , solution_size,weights,greater_is_better=False,T=20,nr=2):
    
        #matrice des distances
    
    dists = vectors_dist(weights)
    
        #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]
    
        #solutions initiales
    
    current_solutions = [[uniform(0,1) for x in range(0,solution_size)] for y in range(len(weights))]
    
        # leurs valeurs
    
    current_solutionsV = [expected_improvement(np.asarray(current_solutions[x]),gaussian_processes[x],evaluated_losses[x],solution_size, greater_is_better) for x in range(len(current_solutions))]
    
        #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        total_eval+=1
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            
            #selection des indexs des parents dans les T voisins les plus proche 
            samples = sample(range(0,T), 2)
            
            newsolution = DE(current_solutions[B[i][0]],current_solutions[B[i][samples[0]]],current_solutions[B[i][samples[1]]])
            newsolution = mutation1(newsolution,1/solution_size)
            newsolutionV = expected_improvement(np.asarray(newsolution),gaussian_processes[i],evaluated_losses[i],solution_size,greater_is_better)
            c = 0
            bi = list(B[i])
            shuffle(bi)
            for index in bi:
                if (newsolutionV < current_solutionsV[index]) & (c < nr):
                    current_solutions[index] = newsolution
                    current_solutionsV[index] = newsolutionV
            
    
    return current_solutions

"""

prbl = ZDT1_problem()

weights = weight_vectors(24) 

    
current_solutions = [[uniform(0,1) for x in range(0,prbl.size)] for y in range(20)]
    
    # leurs valeurs
    
current_solutionsV = [prbl.all_f(current_solutions[x]) for x in range(20)]

values = [weighted_sum(weights[0],current_solutionsV[x]) for x in range(len(current_solutionsV))]

dots = current_solutions
            # Instanciate a Gaussian Process model
            # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor()

            # Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(dots,values)

totest = [uniform(0,1) for x in range (0,prbl.size)]
totest = np.array(totest)
x_to_predict = totest.reshape(-1,2)

print(x_to_predict)
gp.predict(x_to_predict)



totest = [uniform(0,1) for x in range (0,prbl.size)]
totest = np.array(totest)
expected_improvement(totest, gp, values, greater_is_better=False, n_params=1)
"""