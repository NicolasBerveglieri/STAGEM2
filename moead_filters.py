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
from tools import *
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
import itertools


def moead_hybrid(problem,filename,solution_size,weight_size=20,T=6,delta=0.0,nr=2,saves=[10,30,70,100,-1]):
    current_save=0
    save = []
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
   # weights = [[1,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]]
    
    print(weights)

    #sys.exit()
    #matrice des distances
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = copy.copy(current_solutions)
    
    # data values 
    
    data_values = copy.copy(current_solutionsV)
    
    # création et première update de z
    
    #z = z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        
        total_eval+=1
        if total_eval == saves[current_save]:
            """            
            indexes = offline_filter([x.tolist() for x in data_values])
            save_values = [data_values[i].tolist() for i in indexes]
            save_train = [data_train[i] for i in indexes]
            save_values.sort()
            save_train.sort()
            print(save_values)
            save[current_save] = list(save_values for save_values,_ in itertools.groupby(save_values))
            """
            
            save.append(offline_filter([x.tolist() for x in data_values]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1
            

        print(total_eval)
        if total_eval > 40:
            mod1 =  SVR(kernel='rbf',C=30, gamma=0.2)
            mod1.fit(data_train,[data_values[x][0] for x in range(len(data_values))])
            mod2 =  SVR(kernel='rbf',C=30, gamma=0.2)
            mod2.fit(data_train,[data_values[x][1] for x in range(len(data_values))])
        
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            temp = [[],sys.maxint]
            for j in range(0,8):
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
                if total_eval > 40:                
                    values = [mod1.predict([newsolution])[0], mod2.predict([newsolution])[0]]
                else:
                    values = problem(newsolution)
                #print("values estimated:",values)
                #Rvalues = problem(newsolution)
                #print("real values:", Rvalues)
                #print(values - Rvalues, newsolution)
                if (temp[1]> weighted_sum(weights[i],values) ):
                    temp = [newsolution,weighted_sum(weights[i],values)]  
            y = problem(temp[0])
            #z = z_update(z,y)
            data_train.append(temp[0])
            data_values.append(y)
            
            c = 0
            bi = list(B[i])       
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if weighted_sum(weights[index],y) < weighted_sum(weights[index],current_solutionsV[index]):
                        c+=1
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y
    big_save(filename,save,current_solutions,current_solutionsV)
    #save_mono(filename, current_solutions,current_solutionsV)
    return current_solutionsV



def moead_filter(problem,filename,solution_size,weight_size=20,T=6,delta=0.0,nr=2,saves=[10,30,70,100,-1]):
    current_save=0
    save = []
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
   # weights = [[1,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]]
    
    print(weights)

    #sys.exit()
    #matrice des distances
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = copy.copy(current_solutions)
    
    # data values 
    
    data_values = copy.copy(current_solutionsV)
    
    # création et première update de z
    
    #z = z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        print(total_eval)
        total_eval+=1
        if total_eval == saves[current_save]:
            """            
            indexes = offline_filter([x.tolist() for x in data_values])
            save_values = [data_values[i].tolist() for i in indexes]
            save_train = [data_train[i] for i in indexes]
            save_values.sort()
            save_train.sort()
            print(save_values)
            save[current_save] = list(save_values for save_values,_ in itertools.groupby(save_values))
            """
        
            save.append(offline_filter([x.tolist() for x in data_values]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1
            

        mod1 =  SVR(kernel='rbf',C=30, gamma=0.2)
        mod1.fit(data_train,[data_values[x][0] for x in range(len(data_values))])
        mod2 =  SVR(kernel='rbf',C=30, gamma=0.2)
        mod2.fit(data_train,[data_values[x][1] for x in range(len(data_values))])
        
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            temp = [[],sys.maxint]
            for j in range(0,8):
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
                
                #values = [mod1.predict([newsolution])[0], mod2.predict([newsolution])[0]]

                if total_eval <15:
                    values = problem(newsolution)
                else:
                    values = [mod1.predict([newsolution])[0], mod2.predict([newsolution])[0]]

                if (temp[1]> weighted_sum(weights[i],values) ):
                    temp = [newsolution,weighted_sum(weights[i],values)]  
            y = problem(temp[0])
            #z = z_update(z,y)
            data_train.append(temp[0])
            data_values.append(y)
            
            c = 0
            bi = list(B[i])       
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if weighted_sum(weights[index],y) < weighted_sum(weights[index],current_solutionsV[index]):
                        c+=1
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y
    big_save(filename,save,current_solutions,current_solutionsV)
    #save_mono(filename, current_solutions,current_solutionsV)
    return current_solutionsV
    
    
    
def moead(problem,filename,solution_size,weight_size=20,T=6,delta=0.0,nr=2,saves=[10,30,70,100,-1]):
    current_save=0
    save = []
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
   # weights = [[1,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]]
    
    print(weights)

    #sys.exit()
    #matrice des distances
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = copy.copy(current_solutions)
    
    # data values 
    
    data_values = copy.copy(current_solutionsV)
    
    # création et première update de z
    
    #z = z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        print(total_eval)
        total_eval+=1
        if total_eval == saves[current_save]:            
            save.append(offline_filter([x.tolist() for x in data_values]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1
            

        
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            temp = [[],sys.maxint]
            for j in range(0,1):
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
                values = problem(newsolution)
                if (temp[1]> weighted_sum(weights[i],values) ):
                    temp = [newsolution,weighted_sum(weights[i],values)]  
            y = problem(temp[0])
            data_train.append(temp[0])
            data_values.append(y)
            
            c = 0
            bi = list(B[i])       
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if weighted_sum(weights[index],y) < weighted_sum(weights[index],current_solutionsV[index]):
                        c+=1
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y
    big_save(filename,save,current_solutions,current_solutionsV)
    return current_solutionsV
    
    
def moead_omniscient(problem,filename,solution_size,weight_size=20,T=6,delta=0.0,nr=2,saves=[10,30,70,100,-1]):
    current_save=0
    save = []
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
   # weights = [[1,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]]
    
    print(weights)

    #sys.exit()
    #matrice des distances
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = copy.copy(current_solutions)
    
    # data values 
    
    data_values = copy.copy(current_solutionsV)
    
    # création et première update de z
    
    #z = z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        print(total_eval)
        total_eval+=1
        if total_eval == saves[current_save]:
            """            
            indexes = offline_filter([x.tolist() for x in data_values])
            save_values = [data_values[i].tolist() for i in indexes]
            save_train = [data_train[i] for i in indexes]
            save_values.sort()
            save_train.sort()
            print(save_values)
            save[current_save] = list(save_values for save_values,_ in itertools.groupby(save_values))
            """
            
            save.append(offline_filter([x.tolist() for x in data_values]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1
            

        
        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            temp = [[],sys.maxint]
            for j in range(0,8):
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
                values = problem(newsolution)
                if (temp[1]> weighted_sum(weights[i],values) ):
                    temp = [newsolution,weighted_sum(weights[i],values)]  
            y = problem(temp[0])
            #z = z_update(z,y)
            data_train.append(temp[0])
            data_values.append(y)
            
            c = 0
            bi = list(B[i])       
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if weighted_sum(weights[index],y) < weighted_sum(weights[index],current_solutionsV[index]):
                        c+=1
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y
    big_save(filename,save,current_solutions,current_solutionsV)
    #save_mono(filename, current_solutions,current_solutionsV)
    return current_solutionsV
    
    
    
    
    
def moead_substitute(problem,filename,solution_size,weight_size=20,T=6,delta=0.0,nr=2,saves=[10,30,70,100,-1]):
    current_save=0
    save = []
    #creation des vecteurs de poids 
    weights = weight_vectors(weight_size) 
   # weights = [[1,0],[0.9,0.1],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5],[0.4,0.6],[0.3,0.7],[0.2,0.8],[0.1,0.9],[0,1]]
    
    print(weights)

    #sys.exit()
    #matrice des distances
    dists = vectors_dist(weights)
    
    #Matrice des T vecteurs les plus proches pour chaques vecteurs
    
    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #solutions initiales
    
    current_solutions = [[uniform(-100,100) for x in range(0,solution_size)] for y in range(len(weights))]
    
    # leurs valeurs
    
    current_solutionsV = [problem(current_solutions[x]) for x in range(len(current_solutions))]
    
    
    # data training
    
    data_train = copy.copy(current_solutions)
    
    # data values 
    
    data_values = copy.copy(current_solutionsV)
    
    # création et première update de z
    
    #z = z_creation(current_solutionsV)
    
    #debut de la boucle
    total_eval = 0 #len(current_solutions)
    
    while(total_eval < 100):
        total_eval+=1
        if total_eval == saves[current_save]:
            """            
            indexes = offline_filter([x.tolist() for x in data_values])
            save_values = [data_values[i].tolist() for i in indexes]
            save_train = [data_train[i] for i in indexes]
            save_values.sort()
            save_train.sort()
            print(save_values)
            save[current_save] = list(save_values for save_values,_ in itertools.groupby(save_values))
            """
            
            save.append(offline_filter([x.tolist() for x in data_values]))
            save[current_save].sort()
            save[current_save] = list(save[current_save] for save[current_save],_ in itertools.groupby(save[current_save]))
            current_save+=1
            
            
        #le model entre en jeu après 10 generations et apprend toutes les 5 generations
        if  ((10  < total_eval) & (total_eval % 5 == 1)):
            mod1 =  SVR(kernel='rbf',C=30, gamma=0.2)
            mod1.fit(data_train,[data_values[x][0] for x in range(len(data_values))])
            mod2 =  SVR(kernel='rbf',C=30, gamma=0.2)
            mod2.fit(data_train,[data_values[x][1] for x in range(len(data_values))])
        


        # pour chaque vecteur de poids, on va créer des solutions puis en choisir une
        for i in range(len(weights)):
            temp = [[],sys.maxint]
            for j in range(0,1):
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
                if (total_eval < 10 ) or (total_eval % 5 == 0):
                    values = problem(newsolution)
                else:
                    values = [mod1.predict([newsolution])[0], mod2.predict([newsolution])[0]]

                if (temp[1]> weighted_sum(weights[i],values) ):
                    temp = [newsolution,weighted_sum(weights[i],values)]  
            y = problem(temp[0])
            #z = z_update(z,y)
            data_train.append(temp[0])
            data_values.append(y)
            
            c = 0
            bi = list(B[i])       
            shuffle(bi)
            for index in bi:
                if (c < nr):
                    if weighted_sum(weights[index],y) < weighted_sum(weights[index],current_solutionsV[index]):
                        c+=1
                        current_solutions[index] = temp[0]
                        current_solutionsV[index] = y
    big_save(filename,save,current_solutions,current_solutionsV)
    #save_mono(filename, current_solutions,current_solutionsV)
    return current_solutionsV
    
    