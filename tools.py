# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 20:43:20 2018

@author: nicolas
"""
import sys
from random import randint
import random
from random import *
import ast
import matplotlib.pyplot as plt
import os.path
import copy
import numpy as np

def z_creation(solutions):
    z = [sys.maxsize]*len(solutions[0])
    for solution in solutions:
        for i in range(len(solution)):
            tmp = solution[i]
            if (tmp < z[i]):
                z[i] = tmp
    return z
    
    

def z_update(z,solution):
    for i in range(len(solution)):
        if (solution[i] < z[i]):
            z[i] = solution[i]
    return z
        
    
def int_sum(x):
    return sum(range(0,x))
    
    
def weighted_parents(l):
    #select parents
    p11,p22,p33 = -1,-1,-1
    while p11 == p22 or p22 == p33 or p11 == p33:
        p1 = randint(0,int_sum(len(l)))
        p2 = randint(0,int_sum(len(l)))
        p3 = randint(0,int_sum(len(l)))
        #print(p1,p2,p3)
        v1,v2,v3 = False,False,False
        i = 0
        j = 0
        while v1 == False or v2 == False or v3 == False:
            i+=1
            j+=i
            if j > p1 and v1 == False:
                p11 = i
                v1 = True
            if j > p2 and v2 == False:
                p22 = i
                v2 = True
            if j > p3 and v3 == False:
                p33 = i
                v3 = True

    return p11-1,p22-1,p33-1      


def dominate(sol1,sol2,greater_better=False):
    for i in range(len(sol1)):
        if sol1[i] > sol2[i]:
            return False
    return True 

def dominated(sol,listsol,greater_better=False):
    for s in listsol:
        if dominate(s,sol,greater_better) & (s != sol):
            return True
    return False
                        
def offline_filter(solutionsV):
    print("saving progress")
    return [x for x in solutionsV if dominated(x,solutionsV) == False]
    #return [i for i in range(len(solutionsV)) if dominated(solutionsV[i],solutionsV) == False]
    

def save_mono(filename,sol,val):
   file = open(filename,'w')
   file.write('solutions\n')
   for s in sol:
       file.write(str(s)+'\n')
   file.write('values\n')
   for v in val:
       file.write(str(v)+'\n')
   file.close()  

def big_save(filename,saves,endsol,endval):
   file = open(filename,'w')
   for save in saves:
       file.write('values\n')
       for s in save:
           file.write(str(s)+'\n')
   file.write('end_solutions\n')
   for s in endsol:
       file.write(str(s)+'\n')
   file.write('end_values\n')
   for v in endval:
       file.write(str(v)+'\n')
   file.close()
   
   
def read_front(filename):
    file = open(filename,'r')
    pareto_fronts = []
    lines = file.read().splitlines()
    front = []
    for line in lines:
        if line == "end_solutions":
            pareto_fronts.append(front)
            break
        if line == "solutions":
            pareto_fronts.append(front)
            front = []
        elif line == "values":
            if len(front) != 0:
                pareto_fronts.append(front)
            front = []
        else:
            front.append(ast.literal_eval(line))
    file.close()
    return pareto_fronts
            
def read_mono(filename):
    file = open(filename,'r')
    sol = []
    val = []
    part = 1
    lines = file.read().splitlines()
    for line in lines:
        if line == "solutions":
            continue
        if line == "values":
            part = 2
            continue
        if part == 1:
            s = ast.literal_eval(line)
            sol = sol + [s]
        if part == 2:
            v = ast.literal_eval(line)
            val = val + [v]
    return sol,val
    
    
def draw_pareto_front_layers(layers,name,gen=[10,30,70,100,-1],colors=["black","turquoise","yellow",
                             "red","green","fuchsia","blue","olive","burlywood","pink","indigo","slategrey"]):
    cpt = 0
    for layer in layers:
        y = [x[0] for x in layer]
        z = [x[1] for x in layer]
        plt.scatter(y, z, color=colors[cpt], label='generation_'+str(gen[cpt]))
        cpt += 1
    plt.title("MOEAD_ego pareto front on problem 0")
    plt.ylabel('f1')
    plt.xlabel('f2')
    plt.legend(loc=2,fontsize = 'small')
    #plt.savefig(name)
    plt.show()

def save_all_figures(dirname,dirsave):
    for i in range(0,150):
        if os.path.isfile(dirname+"/problem_"+str(i)):
            layers = read_front(dirname+"/problem_"+str(i))
            for j in range(len(layers)):
                layers[j] = [x for x in layers[j] if  (2000 > x[0]) & (2000 > x[1]) & (x[0] > -2000) & (x[1] > -2000)]
            draw_pareto_front_layers(layers,dirsave+"/problem_"+str(i))
    


def max2(points):
    maximum = points[0][1]

    for item in points:
        # Compare elements by their weight stored
        # in their second element.
        if item[1] > maximum:
            maximum = item[1]

    return maximum     
         
def hp(points,ref_point=[0,0]):

    lw = ref_point[0]
    lh = ref_point[1]
    hypervolume = 0
    for p in points:
     hypervolume += (lw - p[0]) * (lh - p[1])
     lh = p[1]
    return hypervolume
            
def hypervolume(filesname,gen):
    layers = [] 
    max_w = -100000
    max_h = -100000
    for file in filesname:
        layers.append(read_front(file))
    for layer in layers:
        tmp1 = max(layer[gen-1])[0]
        tmp2 = max2(layer[gen-1])
        if  tmp1  >  max_w:
            max_w = tmp1
        if   tmp2 >   max_h:
            max_h = tmp2 
            
    ref = [max_w,max_h]
    volumes = [hp(x[gen-1],ref) for x in layers]  
    return volumes            
"""
layers = read_front("MOEAD_FILTER_data/problem_30")  
for i in range(len(layers)):
    layers[i] = [x for x in layers[i] if  (2000 > x[0]) & (2000 > x[1])]
    
draw_pareto_front_layers(layers)
"""
#save_all_figures("MOEAD_OMNISCIENT_data","MOEAD_OMNISCIENT_png")

direc = ["MOEAD_PURE_data/","MOEAD_SUBSTITUTE_data/","MOEAD_FILTER_data/","MOEAD_OMNISCIENT_data/","MOEAD_HYBRID_data/","MOEAD_EGO_data/"]

dim2 = 0
dim3 = 825
dim5 = 1650
dim10 = 2475
dim20 = 3300
dim40 = 4125
first  = [1, 3, 5, 7, 9, 20, 22, 24, 26, 35, 37, 39, 46, 48, 53]
todo = [[(x-1) *15 +d for x in first] for d in [dim2,dim3,dim10,dim20]]
colors=["black","turquoise","yellow","red","green","fuchsia","blue","olive","burlywood","pink","indigo","slategrey"]

def compute_all_hp(direc,todo):
    all_hyp = []
    all_hyp_prct = []
    for dim in todo:
        for i in dim:
            files = [x+"problem_"+str(i) for x in direc]
            hpv = [hypervolume(files,g) for g in [1,2,3,4]]
            all_hyp.append(hpv)
            all_hyp_prct.append([[(v / max(vgen)) for v in vgen] for vgen in hpv])
    return all_hyp_prct,all_hyp


def compute_one_hp(direct,todo,dimension,prb):
    all_hyp = []
    all_hyp_prct = []
    files = [x+"problem_"+str(todo[dimension][prb]) for x in direc]
    hpv = [hypervolume(files,g) for g in [1,2,3,4]]
    hpv_prct = [[(v / max(vgen)) for v in vgen] for vgen in hpv]
    print(files)
    return hpv_prct, hpv
    
def graph_bar(values=[[4, 9, 2,3],[1,2,3,4],[10,11,12,13]],colors=colors,algos=  ["MOEAD","MOEAD_SUBSTITUTE","MOEAD_FILTER","MOEAD_OMNISCIENT","MOEAD_HYBRID","MOEAD_EGO"]):
    N = 4
    values = np.array(values)
    values = values.transpose()
    values = values.tolist()
    print(values)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.1       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects = []    
    for i in range(len(values)):
        rects.append(ax.bar(ind+width*i,values[i],width,color=colors[i]))

    ax.set_ylabel('avrg % from best')
    ax.set_xlabel('Number of feature')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('2', '3', '10',"20") )
    
    ax.legend( [rects[x][0] for x in range(len(values))],  algos,loc = 3,fontsize = 'small')
    
    
def candidate_sols(quantity,treshold):
    plt.scatter(quantity,treshold)
    plt.plot(quantity,treshold)
    plt.ylabel('generations')
    plt.xlabel('nbr of candidates')
    plt.show()
    
"""    
quantity = [1,2,4,6,8,10,12,14,16]
treshold = [30,17,15,16,14,12,11,11,11]
treshold2 = [60,46,42,37,32,28,25,24,23]
treshold2 = [x+uniform(-2,2) for x in treshold2]
   
candidate_sols(quantity,treshold2)

#v = compute_one_hp(direc,todo,0,0)[0][3] 
values = [compute_one_hp(direc,todo,3,x)[0][3] for x in range(len(first))]
values = np.array(values)
values = values.transpose()
values = values.tolist()
print([sum(i) / len(i) for i in values])








problem_specification = [[0.7419553087709173, 0.7844215321581759, 0.8419822485698062, 0.9592933084351074, 0.870672451641463, 0.8838877252357764],
           [0.692014213156561, 0.6768373104666224, 0.8957275927597149, 0.9859984001968868, 0.8543143786176229, 0.8846248048551603],
        [0.57270704578600753, 0.6758664067832799, 0.7386753527084292, 0.9121318779502485, 0.9049232444745634, 0.6925347025372386],
        [0.5326921289013459, 0.6331609195175324, 0.6521522924982872, 0.956316168262168, 0.7809986428248156, 0.619665144588622],
        [0.4495413563090936, 0.4493210425402296, 0.7642200804420676, 0.9687686304787948, 0.8038617452726675, 0.4734034930361855],
        [0.367965510856817, 0.2834250535463928, 0.5343089795076632, 0.9772872044121747, 0.6589046229150758, 0.2857261797081256]]
        
        
graph_bar(toplot)
"""

feature_specification = [[0.874872893841324, 0.9058802414768075, 0.8900673609663816, 0.9442083960382618, 0.9088761568114566, 0.9180621799678629],
                         [0.7440575308899927, 0.6894626974723299, 0.8723797011881301, 0.9742804097041993, 0.9547190197869904, 0.8369647153809852],
                        [0.6673469294564958, 0.1842701953195297, 0.6019935682396734, 0.9084394973185228, 0.7551112880907232, 0.0582134837387187],
                        [0.69103189502890043, 0.1469761251777168, 0.4573218066302511, 0.9582912211712975, 0.5558904177458312, 0.0588158274602599],
                       ]

#graph_bar(feature_specification)

#draw_pareto_front_layers(read_front(direc[5]+"problem_0"),"wadup")