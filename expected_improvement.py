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
from tools import *

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

    return  expected_improvement



def expected_improvement_EGO(x, gaussian_process,weight, evaluated_loss, n_params ,greater_is_better=False):
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
    mu1,sigma1 = gaussian_process[0].predict(x_to_predict, return_std=True)
    mu2,sigma2 = gaussian_process[1].predict(x_to_predict, return_std=True)
    mu, sigma = (mu1 * weight[0] + mu2 * weight[1]) , ( sigma1 * weight[0] + sigma2 * weight[1])
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

    return  expected_improvement


def moead_EI(gaussian_processes , evaluated_losses , solution_size,weights,greater_is_better=False,T=6,nr=2):
    
        #matrice des distances
    
    dists = vectors_dist(weights)
    
        #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]
    
        #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
        # leurs valeurs
    
    
        
    current_solutionsV = [expected_improvement_EGO(np.asarray(current_solutions[x]),gaussian_processes,weights[x],evaluated_losses[x],solution_size,greater_is_better) for x in range(len(current_solutions))]   
        
        
        
        #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 40):
        print("MOEAD:"+str(total_eval))
        total_eval+=1
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            diff = False
            unluck = 0
            while (diff == False) and (unluck <20):
               unluck+=1
               samples = sample(range(0,T), 2)
               if current_solutions[B[i][samples[0]]] != current_solutions[B[i][samples[1]]]:
                   diff = True
               if unluck > 2:
                   print("stuck")
                   
    
            newsolution = mutation1(DE(current_solutions[B[i][0]],current_solutions[B[i][samples[0]]],current_solutions[B[i][samples[1]]]),1/solution_size)
            """   
            #selection des indexs des parents dans les T voisins les plus proche 
            samples = sample(range(0,T), 2)
            
            newsolution = DE(current_solutions[B[i][0]],current_solutions[B[i][samples[0]]],current_solutions[B[i][samples[1]]])
            newsolution = mutation1(newsolution,1/solution_size)
            """
            newsolutionV = expected_improvement_EGO(np.asarray(newsolution),gaussian_processes,weights[i], evaluated_losses[i],solution_size,greater_is_better)
            
            if (newsolutionV > current_solutionsV[i]):
                current_solutions[i] = newsolution
                current_solutionsV[i] = newsolutionV
    
    return current_solutions


def evolutionary_EI(gaussian_process, evaluated_loss, solution_size,initial_solution=25, greater_is_better=False):
        
    #solutions initiales
        
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(initial_solution)]
    
     # leurs valeurs
    current_solutionsV = [(expected_improvement(np.asarray(current_solutions[x]),gaussian_process,evaluated_loss,solution_size, greater_is_better)+gaussian_process.predict(np.asarray(current_solutions[x]).reshape(-1,2))) for x in range(len(current_solutions))]
    
    total_eval = 0
    
    while(total_eval < 200):
        total_eval+=1

        #sort parents from worst to best

        indexes = sorted(range(len(current_solutionsV)), key=lambda k: current_solutionsV[k])
        current_solutions = [current_solutions[i] for i in indexes]
        current_solutionsV = [current_solutionsV[i] for i in indexes]
        
        p1_i,p2_i,p3_i = weighted_parents(indexes)
        p1 = current_solutions[p1_i]
        p2 = current_solutions[p2_i]
        p3 = current_solutions[p3_i]
        newsolution = mutation1(DE(p1,p2,p3),1/solution_size)
        newsolutionV = expected_improvement(np.asarray(newsolution),gaussian_process,evaluated_loss,solution_size,greater_is_better)+gaussian_process.predict(np.asarray(newsolution).reshape(-1,2))
        current_solutions += [newsolution]
        current_solutionsV += [newsolutionV]
    indexes = sorted(range(len(current_solutionsV)), key=lambda k: current_solutionsV[k])
    current_solutions = [current_solutions[i] for i in indexes]
    current_solutionsV = [current_solutionsV[i] for i in indexes]
    return current_solutions[len(indexes)-1]

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