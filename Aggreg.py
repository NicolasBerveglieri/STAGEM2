# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:50:24 2018

@author: nicol
"""

def Tchebycheff(weights,values,z):
    return max([weights[x] * abs(z[x] -values[x]) for x in range(len(weights)) ])

def weighted_sum(weights,values):
    return sum([weights[x] * values[x] for x in range(len(weights))])


