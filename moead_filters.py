# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 12:18:39 2018

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
from ZDT1_problem import *
from evo_op import *
from Aggreg import *

def algo(weight_size=100,T=20,delta=0.0,solution_size=10,problem="ZDT1",nr=2):
    
    #instanciation du probleme
    
    prbl = ZDT1_problem()
    
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
    
    print(weights)

    
    #matrice des distances
    
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]

    #solutions initiales
    
    current_solutions = [[uniform(0,1) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [prbl.all_f(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = current_solutions
    
    # data values 
    
    data_values = current_solutionsV
    
    # création et première update de z
    
    z = prbl.z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 200):
        total_eval += 1
        print(total_eval)
        f1_dots = [data_values[x][0] for x in range(len(data_values))]
        f2_dots = [data_values[x][1] for x in range(len(data_values))]
        mod1 = SVR(C=1.0, epsilon=0.2)
        mod1.fit(data_train,f1_dots)
        mod2 = SVR(C=1.0, epsilon=0.2)
        mod2.fit(data_train,f2_dots)
            
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            
            # temp[0] = solution, temp[1] = tchebycheff
            
            temp = [[],10000]
            for j in range(0,16):
                samples = sample(range(0,T), 2)
                newsolution = DE(current_solutions[B[i][0]],current_solutions[B[i][samples[0]]],current_solutions[B[i][samples[1]]])
                newsolution = mutation1(newsolution,1/solution_size)
                values = [mod1.predict([newsolution])[0], mod2.predict([newsolution])[0]]
                if (temp[1] > Tchebycheff(weights[i],values,z)):
                    temp = [newsolution, Tchebycheff(weights[i],values,z)]
            y = prbl.all_f(temp[0])
            z = prbl.z_update(z,y)
            data_train+[temp[0]]
            data_values+[y]
            
            c = 0
            bi = list(B[i])
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if temp[1] < Tchebycheff(weights[index],current_solutions[index],z):
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y

    return current_solutionsV


Vs = algo()
for i in Vs:
    print(i)