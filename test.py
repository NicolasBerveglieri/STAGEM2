# -*- coding: utf-8 -*-
"""
Created on Thu May 31 13:31:29 2018

@author: nicolas
"""

import scipy.optimize
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
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
import copy
import expected_improvement as EI
from mpl_toolkits.mplot3d import Axes3D
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
import time
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)
def funct(vector):
    return sum(vector)
    
def f(x):
    """The function to predict."""
    return np.sin(x) 

def gaussian_test():   
    d = [uniform(0, 11) for x in range(10)]  
    learned_point = copy.copy(d)
    learned_pointV = [f(x) for x in learned_point]
    v = [f(x) for x in d]
    gp =GaussianProcessRegressor()
    
    d = [[x] for x in d]
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(d, v)
    
    
    d = [x/1000 for x in range(0,10000)]
    real = [f(i) for i in d]
    imp = [EI.expected_improvement(np.asarray(p),gp,learned_pointV,1,greater_is_better=False) for p in d]
    d = [[x] for x in d]
    
    pred,stig = gp.predict(d,return_std=True)
    
    pEI = [imp[i]+pred[i] for i in range(0,10000)]
    
    s=[0.1 for x in range(0,10000)]
    plt.scatter(d,real,color="black",label="true",s=s)
    plt.scatter(d, pred, color="red", label="gpr",s=s)
    plt.scatter(d,imp,color="grey",label="EI",s=s)
    plt.scatter(d,pEI,color="yellow",label="pEI",s=s)
    plt.scatter(learned_point,learned_pointV,color="turquoise",label="learning points")
    d = np.array(d)[:,0]
    plt.fill_between(d,pred+stig,pred-stig,color="darkorange",alpha=0.2)
    #plt.legend()
    plt.show()


    
    
def surface_test(problem,name):    
     
    #apprentissage
    svr =  SVR(kernel='rbf',C=100, gamma=0.05)
    points = [[uniform(-20,20) for x in range(0,problem.number_of_variables)] for y in range(0,3000)]    
    values = [problem(x)[0] for x in points]
    svr.fit(points,values)
    print("learning done")
    x = np.outer(np.linspace(-20, 20, 400), np.ones(400))
    y = x.copy().T
    #z = [[problem([x[i][j],y[i][j],0])[0] for j in range(len(x))] for i in range(len(x))]
    #print(svr.predict([[x[0][0],y[0][0],0]]))[0]
    z = [[svr.predict([[x[i][j],y[i][j]]])[0] for j in range(len(x))] for i in range(len(x))]
    z = np.array(z)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
    plt.savefig("fsurface/"+name)
    plt.close()
    
def surface_test_mono(problem,name):    
    
    x = np.outer(np.linspace(-100, 100, 400), np.ones(400))
    y = x.copy().T
    z = [[problem([x[i][j],y[i][j]]) for j in range(len(x))] for i in range(len(x))]
    z = np.array(z)
    #z = [x[i]+y[i] for i in range(len(x))]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(x, y, z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
    plt.savefig("fsurface/"+name)
    plt.close()


def report_pareto_front():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    f1 = [x for x in range(0,20)]
    f2 = [20-x for x in range(0,20)]
    f3 = [randint(0,20) for x in range(0,20)]
    ax.scatter(f1,f2,f3,c="red",zdir="y")
    d1 = [x -randint(5,10) for x in f1]
    d2 = [x -randint(5,10) for x in f2]
    d3 = [x -randint(5,10) for x in f3]
    ax.scatter(d1,d2,d3,c="black",zdir="y")


def report_model_noise_noiseless():
    d = [5.132495750645059, 6.890631085568639, 8.037130112622666, 5.730558203430376, 6.1606160212828245, 1.99880496869834, 9.184545431887996, 6.243178208010988, 0.5818207295235234, 8.017023747911812]
#[uniform(0, 10) for x in range(10)]  
    learned_point = copy.copy(d)
    learned_pointV = [f(x) for x in learned_point]
    learned_pointV = [x+ uniform(-1,1) for x in learned_pointV]
    v = [f(x) for x in d]
    gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    + WhiteKernel(1e-1)
    gp =GaussianProcessRegressor(kernel = gp_kernel)
    svr =  SVR(kernel='rbf',C=50, gamma=0.1)
    d = [[x] for x in d]
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(d, learned_pointV)
    svr.fit(d,learned_pointV)
    
    d = [x/100 for x in range(0,1000)]
    real = [f(i) for i in d]
    imp = [EI.expected_improvement(np.asarray(p),gp,learned_pointV,1,greater_is_better=True) for p in d]
    d = [[x] for x in d]
    
    pred,stig = gp.predict(d,return_std=True)
    pred_svr = svr.predict(d)
    pEI = [imp[i]+pred[i] for i in range(0,1000)]
    
    s=[0.1 for x in range(0,1000)]
    plt.scatter(d,real,color="black",label="true",s=s)
    plt.scatter(d, pred, color="red", label="gpr",s=s)
    plt.scatter(d,pred_svr,color="green",label="svr",s=s)
    plt.scatter(learned_point,learned_pointV,color="turquoise",label="learning points")
    d = np.array(d)[:,0]
    plt.fill_between(d,pred+stig,pred-stig,color="darkorange",alpha=0.2)
    #plt.legend()
    plt.show()

def report_weight_vector():
    for i in range(0,21):
            x = 0 + 1 * np.cos((np.pi/40)*i)
            y = 0 + 1 * np.sin((np.pi/40)*i)
            plt.plot([0,x],[0,y],"turquoise")
    plt.axis('square')
    plt.ylabel('objective 1')
    plt.xlabel('objective 2')
    #plt.axis('off')
    plt.show()

def report_pareto_set_learning():
    d = [uniform(0, 5) for x in range(100)] + [uniform(5,10) for x in range(5)]  
    learned_point = copy.copy(d)
    learned_pointV = [f(x) for x in learned_point]
    v = [f(x) for x in d]
    svr =  SVR(kernel='rbf',C=50, gamma=0.1)
    d = [[x] for x in d]
    # Fit to data using Maximum Likelihood Estimation of the parameters
    svr.fit(d,learned_pointV)
    
    d = [x/100 for x in range(0,1000)]
    real = [f(i) for i in d]
    d = [[x] for x in d]
    
    pred_svr = svr.predict(d)
    
    s=[0.1 for x in range(0,1000)]
    plt.scatter(d,real,color="black",label="true",s=s)
    plt.scatter(d,pred_svr,color="red",label="svr",s=s)
    plt.scatter(learned_point,learned_pointV,color="turquoise",label="learning points")
    plt.legend(loc=2, prop={'size': 8})
    plt.show()


def report_expected_improvement():
    d = [uniform(0, 11) for x in range(8)]  
    learned_point = copy.copy(d)
    learned_pointV = [f(x) for x in learned_point]
    v = [f(x) for x in d]
    gp =GaussianProcessRegressor()
    
    d = [[x] for x in d]
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(d, v)
    
    
    d = [x/100 for x in range(0,1000)]
    real = [f(i) for i in d]
    imp = [EI.expected_improvement(np.asarray(p),gp,learned_pointV,1,greater_is_better=True) for p in d]
    d = [[x] for x in d]
    
    pred,stig = gp.predict(d,return_std=True)
    
    pEI = [imp[i]+pred[i] for i in range(0,1000)]
    
    s=[0.1 for x in range(0,100)]
    plt.scatter(d,real,color="black",label="true",s=s)
    plt.scatter(d, pred, color="red", label="gpr",s=s)
    plt.scatter(d,imp,color="green",label="EI",s=s)
    plt.scatter(learned_point,learned_pointV,color="turquoise",label="learning points")
    d = np.array(d)[:,0]
    plt.fill_between(d,pred+stig,pred-stig,color="darkorange",alpha=0.2)
    plt.legend(loc=3, prop={'size': 8})

    plt.show()

def test_models(problem):
    points = [[uniform(-20,20) for x in range(0,problem.number_of_variables)] for y in range(0,1000)]
    topredict = [[uniform(-20,20) for x in range(0,problem.number_of_variables)] for y  in range(0,10000)]
    real = [problem(x)[0] for x in topredict]
    values = [problem(x)[0] for x in points]
    
    gp_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp =GaussianProcessRegressor(kernel =gp_kernel)
    svr =  SVR(kernel='rbf',C=200, gamma=0.01)
    kr = KernelRidge(kernel='rbf', gamma=0.1)
    
    
    ctime = time.clock()
    gp.fit(points,values)
    gp_learningtime = time.clock() - ctime
    ctime = time.clock()    
    svr.fit(points,values)
    svr_learningtime = time.clock() - ctime
    ctime = time.clock() 
    kr.fit(points,values)
    kr_learningtime = time.clock() - ctime
    models = [gp,svr,kr]
    
    
    ctime = time.clock() 
    predictions_gp =  [gp.predict([x])[0] for x in topredict] 
    gp_predictiontime = time.clock() - ctime 
    ctime = time.clock() 
    predictions_svr =  [svr.predict([x])[0] for x in topredict] 
    svr_predictiontime = time.clock() - ctime
    ctime = time.clock() 
    predictions_kr =  [kr.predict([x])[0] for x in topredict] 
    kr_predictiontime = time.clock() - ctime
    
        
    gp_dist = [abs(predictions_gp[i] - real[i]) for i in range(len(real))]
    gp_dist_prc = [gp_dist[i]/ real[i] for i in range(len(real)) ]
    svr_dist = [abs(predictions_svr[i] - real[i]) for i in range(len(real))]
    svr_dist_prc = [svr_dist[i]/ real[i] for i in range(len(real)) ]
    kr_dist = [abs(predictions_kr[i] - real[i]) for i in range(len(real))]
    kr_dist_prc = [kr_dist[i]/ real[i] for i in range(len(real)) ]
    print("PREDDDD",predictions_gp[0],real[0])
    gp_avg =  sum(gp_dist_prc) /float(len(gp_dist_prc))
    svr_avg =  sum(svr_dist_prc) /float(len(svr_dist_prc))
    kr_avg =  sum(kr_dist_prc) /float(len(kr_dist_prc))    
    
    gp_max_dist = gp_dist.index(max(gp_dist))
    svr_max_dist = svr_dist.index(max(svr_dist))
    kr_max_dist = kr_dist.index(max(kr_dist)) 
    
    
    gp_max_dist_prc = gp_dist[gp_max_dist] / real[gp_max_dist]
    svr_max_dist_prc = svr_dist[svr_max_dist] / real[svr_max_dist]    
    kr_max_dist_prc = kr_dist[kr_max_dist] / real[kr_max_dist]    
    
    
    print(gp_learningtime,svr_learningtime,kr_learningtime,gp_predictiontime,svr_predictiontime,kr_predictiontime)
    print(gp_avg,svr_avg,kr_avg,gp_max_dist_prc,svr_max_dist_prc,kr_max_dist_prc,max(gp_dist),max(svr_dist),max(kr_dist))
    
    
def test_gaussian(problem):
    points = [[uniform(-20,20) for x in range(0,problem.number_of_variables)] for y in range(0,100)]
    topredict = [[uniform(-20,20) for x in range(0,problem.number_of_variables)] for y  in range(0,1000)]
    real = [problem(x)[0] for x in topredict]
    values = [problem(x)[0] for x in points]
    
    gp_kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
    gp =GaussianProcessRegressor(kernel =gp_kernel)    
    ctime = time.clock()
    gp.fit(points,values)
    gp_learningtime = time.clock() - ctime
    ctime = time.clock() 
    [gp.predict([x],return_std=True) for x in topredict] 
    gp_predictiontime = time.clock() - ctime 
    print(gp_predictiontime)
    ctime = time.clock() 
    [gp.predict([x]) for x in topredict] 
    gp_predictiontime = time.clock() - ctime
    print(gp_predictiontime)
    
    
    
report_expected_improvement()