# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy.linalg as np
from scipy.spatial import distance
import heapq
from random import randint
import random
import sys
import sklearn
from sklearn.svm import SVR
import numpy as np



def random_mubqp(n,m,filename,r):
    file = open(filename,'w')
    file.write('c\n')
    file.write('c wadup\n')
    file.write('c\n')
    file.write('p\n')
    file.write('p MUBQP 0 '+str(m)+' '+str(n)+' '+str(r)+'\n')
    for i in range(0,n*n):
        s = ''
        s2= ''
        for j in range(m):
            s+= str(random.randint(-100,100))
            s2+= '0'
            if j != (m-1):
                s+= ' '
                s2+= ' '
        s+='\n'
        s2+='\n'
        if (random.random() < r):
            file.write(s2)
        else:
            file.write(s)
    print('done')
    file.close()


def load_data(filename):
    data = open(filename,"r")
    p = 0
    M = 0
    N = 0
    d = 0
    cpt = 0
    pdone = False
    for line in data:
        if line[0] == 'c':
            pass
        elif (line[0] == 'p'):
            if (pdone == False):
                pdone = True
                info = line.split(' ')
                p = info[2]
                M = int(info[3])
                N = int(info[4])
                d = info[5]
                Matrixs = [[[0 for x in range(0,N)] for y in range(0,N)] for z in range(0,M)]
        else:
            rline = line.split()
            for i in range(len(rline)):
                Matrixs[i][cpt // N][cpt % N] = int(rline[i])
            cpt+=1
    return Matrixs

Matrixs = load_data("mydata.dat")

def Tchebycheff(data,z,weights,solution):
    print("mubqp tche")
    return max([weights[x] * abs(z[x] -evalMubqpMono(data,solution,x)) for x in range(len(weights)) ])

def dominate(solutionA,solutionB):
    #est ce que A domine B? (Domine >=)
    for i in range(len(solutionA)):
        if solutionA[i] < solutionB[i]:
            return False
    return True


def evalMubqp(data,solution,weights):
    tmp = [[data[w][i][j]*solution[i]*solution[j] for i in range(len(data[0][0])) for j in range(len(data[0][0]))] for w in range(len(weights))]
    return sum([sum(tmp[x])*weights[x] for x in range(len(weights))])

def evalMubqpNoWeight(data,solution):
    tmp = [[data[w][i][j]*solution[i]*solution[j] for i in range(len(data[0][0])) for j in range(len(data[0][0]))] for w in range(len(data))]
    return [sum(tmp[x]) for x in range(len(data))]

def evalMubqpMono(data,solution,objectif):
    return sum([data[objectif][i][j]*solution[i]*solution[j] for i in range(len(data[0][0])) for j in range(len(data[0][0]))])

def bestObjectiveMubqp(data,solutions):
    z = [-sys.maxsize]*len(data)
    for solution in solutions:
        for i in range(len(data)):
            tmp = evalMubqpMono(data,solution,i)
            if (tmp > z[i]):
                z[i] = tmp
    return z


def weight_vectors(n=10):
    weights = []
    for i in range (0,n+1):
        weights.append([i/n,(n-i) / n])
    return weights



def vectors_dist(weights):
    return [[distance.euclidean(weights[x],weights[y]) for x in range(0,len(weights))] for y in range(0,len(weights))]


def childSolution(parentA, parentB):
    return [parentA[x] if  random.randint(0,1) ==1 else parentB[x] for x in range(len(parentA)) ]

def meoad(data,weights,T=4):

    #1.1

    EP = []

    #1.3

    current_solutions = [[random.randint(0, 1) for x in range(0,len(Matrixs[0][0]))] for y in range(len(weights))]
    current_solutionsV = [evalMubqpNoWeight(data,current_solutions[x]) for x in range(len(current_solutions))]

    print(current_solutionsV)

    #1.4

    z = bestObjectiveMubqp(data,current_solutions)


    #1.2

    dists = vectors_dist(weights)

    B = [heapq.nsmallest(T,range(len(dists)),dists[x].__getitem__) for x in range(len(dists))]


    #2


    for stopCrit in range(0,1000):
        for i in range(len(weights)):

          #2.1

            sample = random.sample(range(0,T), 2)
            x = sample[0]
            l = sample[1]
            child = childSolution(current_solutions[B[i][x]],current_solutions[B[i][l]])

            #2.2

            #2.3

            for j in range(len(weights[0])):
                objJValue = evalMubqpMono(data,child,j)
                if objJValue > z[j]:
                    z[j] = objJValue

            #2.4

            for index in B[i]:
                if Tchebycheff(data,z,weights[index],child) < Tchebycheff(data,z,weights[index],current_solutions[index]):
                    current_solutions[index] = child
                    current_solutionsV[index] =  evalMubqpNoWeight(data,child)

            #2.5
            dominated = False
            for vec in EP:
                if dominate(current_solutionsV[index],vec):
                    EP.remove(vec)
                elif dominate(vec,current_solutionsV[index]):
                    dominated = True
                    break
            if len(EP) == 0 or not(dominated):
                EP.append(current_solutionsV[index])
            #3

    return EP

#res = meoad(Matrixs,weight_vectors(10))
#[print(res[x]) for x in range(len(res))]
#random_mubqp(10,2,"mydata.dat",0.3)


print('The scikit-learn version is {}.'.format(sklearn.__version__))
"""
n_samples, n_features = 10, 2
np.random.seed(0)
y = np.random.randn(n_samples)
weights = [x/10 for x in range(0,10)]
X = np.random.randn(n_samples, n_features)

print(y)
print(X)
print(weights)
clf = SVR(C=1.0, epsilon=0.2)
clf.fit(X,y,weights)"""
