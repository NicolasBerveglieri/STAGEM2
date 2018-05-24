# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 11:22:10 2018

@author: nicol
"""

import math
import numpy.linalg as np
from scipy.spatial import distance
import heapq
from random import randint
import random
import sys
import sklearn
from sklearn.svm import SVR
import numpy as np


class ZDT1_problem:
    
    def __init__( self ):
       self.size = 2

    def f1(x):
        return x[0]
    
    
    def g(x):
       return 1+ 9 *  (sum(x) - x[0]) / (len(x) -1)
    
    
    def f2(self,x):
        return ZDT1_problem.g(x) * (   1- math.sqrt(  ZDT1_problem.f1(x) / ZDT1_problem.g(x)  )    )
    
    def all_f(self,x):
        return ZDT1_problem.f1(x), ZDT1_problem.f2(self,x)
    
    def z_creation(self,solutions):
        z = [sys.maxsize]*len(solutions[0])
        for solution in solutions:
            for i in range(len(solution)):
                tmp = solution[i]
                if (tmp < z[i]):
                    z[i] = tmp
        return z
    
    
    def z_update(self,z,solution):
        for i in range(len(solution)):
            if (solution[i] < z[i]):
                z[i] = solution[i]
        return z
        
    