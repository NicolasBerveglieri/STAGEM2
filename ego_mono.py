# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:24:40 2018

@author: nicolas
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
from ZDT1_problem import *
from evo_op import *
from Aggreg import *
from tools import *
from sklearn.gaussian_process import GaussianProcessRegressor
from expected_improvement import *

def EGO_mono(problem,filename,solution_size=10,solution_initial=75):
    
    print("debut ego_mono")    
    
    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(solution_initial)]
    
        
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
        
    #debut de la boucle
    total_eval = 0  
    gp = GaussianProcessRegressor()
    while(total_eval < 200):
        print("nombre d'evaluations: ",total_eval)
        total_eval+=1
        dots = current_solutions
        # Instanciate a Gaussian Process model
        # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

        # Fit to data using Maximum Likelihood Estimation of the parameters       
        gp.fit(dots,current_solutionsV)

        # find new solution with evolutionary algorithm on expected improvement            
            
        newsolution = evolutionary_EI(gp, current_solutionsV, solution_size)
        newsolutionV = problem(newsolution)
        current_solutions += [newsolution]
        current_solutionsV += [newsolutionV]
        save_mono(filename,current_solutions,current_solutionsV)
    return current_solutions, current_solutionsV