# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:49:04 2018

@author: nicol
"""

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
from expected_improvement import *
from ZDT1_problem import *
from evo_op import *
from Aggreg import *
from sklearn.gaussian_process import GaussianProcessRegressor


def algo(problem,weight_size=24,init_pop=100,update_pop=24):
            
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
    
    #génération des premières solutions
    
    current_solutions = [[uniform(0,1) for x in range(0,problem.size)] for y in range(init_pop)]
    
    # leurs valeurs
    
    current_solutionsV = [problem.all_f(current_solutions[x]) for x in range(init_pop)]
    
    
        #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 1):
        total_eval+= 1
        #création des gaussians process 
        
        models = []
        valuesS = []
        for i in range(len(weights)):
            values = [weighted_sum(weights[i],current_solutionsV[x]) for x in range(len(current_solutionsV))]
            dots = current_solutions
            # Instanciate a Gaussian Process model
            # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor()

            # Fit to data using Maximum Likelihood Estimation of the parameters
            gp.fit(dots,values)
            models += [gp]
            valuesS += [values]
        newsol = moead_EI(models , valuesS ,problem.size,weights)
        newsolV = [problem.all_f(newsol[x]) for x in range(len(newsol))]
        current_solutions += newsol
        current_solutionsV += newsolV
    
    return current_solutions, current_solutionsV
        

        
prbl = ZDT1_problem()
algo(prbl)