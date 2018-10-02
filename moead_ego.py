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
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
import itertools



def moead_EGO(problem,filename,weight_size=24,init_pop=30,update_pop=24,saves=[10,30,70,100,-1]):
    current_save=0
    save = []         
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
    solution_size=problem.number_of_variables
    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    gp_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 20):
        print(total_eval)
        total_eval+= 1
        if total_eval == saves[current_save]:
            
            save.append(offline_filter([x.tolist() for x in current_solutionsV]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1



        gp1 = GaussianProcessRegressor(kernel =gp_kernel)
        gp2 = GaussianProcessRegressor(kernel =gp_kernel)
        # Fit to data using Maximum Likelihood Estimation of the parameters
        if len(current_solutions) < 231:
            gp1.fit(current_solutions,[current_solutionsV[x][0] for x in range(len(current_solutions))])
            gp2.fit(current_solutions,[current_solutionsV[x][1] for x in range(len(current_solutions))])            
        else:
            print("MAX SOLUTION")
            sol_fit = current_solutions[-200:]
            solV_fit = current_solutionsV[-200:]
            gp1.fit(sol_fit,[solV_fit[x][0] for x in range(len(sol_fit))])
            gp2.fit(sol_fit,[solV_fit[x][1] for x in range(len(sol_fit))])            

            
        #création des gaussians process 
        
        valuesS = []
        for i in range(len(weights)):
            values = [weighted_sum(weights[i],current_solutionsV[x]) for x in range(len(current_solutionsV))]
            # Instanciate a Gaussian Process model
            # kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            valuesS += [values]
        newsol = moead_EI([gp1,gp2] , valuesS ,problem.number_of_variables,weights)
        newsolV = [problem(newsol[x]) for x in range(len(newsol))]
        current_solutions += newsol
        current_solutionsV += newsolV
    
    big_save(filename,save,current_solutions,current_solutionsV)

    return current_solutions, current_solutionsV
        

        