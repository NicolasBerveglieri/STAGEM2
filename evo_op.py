# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:25:22 2018

@author: nicol
"""
from random import *

def mean(s1,s2):
    return [(s1[x]+s2[x])/2 for x in range(len(s1))]
    
def random(s1,s2):
    return [s1[x] if randint(0,1) ==1 else s2[x] for x in range(len(s1))]
    
def half(s1,s2):
    return [s1[x] if x < len(s1)/2 else s2[x] for x in range(len(s1))]

def completerandom(s1,s2):
    return [uniform(0,1) for x in range(len(s1))]

def DE(s1,s2,s3,F=0.5,CR=1):
    s = []
    for x in range(len(s1)):
        if uniform(0,1) < CR:
            s = s + [(s1[x] + F * (s2[x] - s3[x]))]
        else:
            s = s + [s1[x]]
    return repair(s)
            

def repair(s):
    return [0 if x < 0 else 1 if x>1  else x for x in s]


def mutation1(s,rate,n=20):     
    rand = uniform(0,1)
    return repair([s[x] if rand > rate else s[x] + sigma(n) * (1 - 0) for x in range(len(s)) ])
    
def sigma(n):
    rand = uniform(0,1)
    sigma = 0
    if rand < 0.5:
        sigma = pow(2 * rand, 1/(n+1)) -1
    else:
        sigma = 1 - pow (2 - 2 * rand, 1/(n-1))           
    return sigma

def any_op(s1,s2):
    r = randint(0,3)
    if (r == 0):
        return mean(s1,s2)
    elif(r == 1):
        return random(s1,s2)
    elif(r == 2):
        return half(s1,s2)
    elif(r == 3):
        return completerandom(s1,s2)



