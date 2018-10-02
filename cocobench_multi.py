#!/usr/bin/env python
"""A short and simple example experiment with restarts.

The code is fully functional but mainly emphasises on readability.
Hence it neither produces any progress messages (which can be very
annoying for long experiments) nor provides batch distribution,
as `example_experiment.py` does.

To apply the code to a different solver, `fmin` must be re-assigned or
re-defined accordingly. For example, using `cma.fmin` instead of
`scipy.optimize.fmin` can be done like::

    import cma
    def fmin(fun, x0):
        return cma.fmin(fun, x0, 2, {'verbose':-9})

"""
from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
import sys
from moead_filters import *
import ego_mono
import test
from moead_ego import *

def testfun(fun,x0):
	size = len(x0)
 
 
d1 = "MOEAD_FILTER"+"_data/"
d2 = "MOEAD_PURE"+"_data/"
d3 = "MOEAD_OMNISCIENT"+"_data/"
d4 = "MOEAD_SUBSTITUTE"+"_data/"
d5 = "MOEAD_HYBRID"+"_data/"
d6 = "MOEAD_EGO"+"_data/"
testdir = "testdir"
	
### input
suite_name = "bbob-biobj"
output_folder = testdir #d6[:-6]
fmin = scipy.optimize.fmin
budget_multiplier = 1  # increase to 10, 100, ...

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

data_directory = testdir #d6  

### go

compteur = 0
cpt2 = 0


dim2 = 0
dim3 = 825
dim5 = 1650
dim10 = 2475
dim20 = 3300
dim40 = 4125
first  = [1, 3, 5, 7, 9, 20, 22, 24, 26, 35, 37, 39, 46, 48, 53]
todo = [[(x-1) *15 +d for x in first] for d in [dim2,dim3,dim10,dim20]]


problem = suite[dim2]
test.test_gaussian(problem)
sys.exit()
print(todo)
for dim in todo:  # this loop will take several minutes or longer
    for index in dim:
        print("Problem number"+str(index)+" starting")
        problem = suite[index]
        direc = data_directory+"problem_"+str(index)
        problem.observe_with(observer)        # generates the data for cocopp post-processing
        moead_substitute(problem,filename=direc,solution_size=problem.number_of_variables,weight_size=30)  # here we assume that the function evaluates the final/returned solution	
        #moead_EGO(problem,filename=direc,weight_size=30,init_pop=30,update_pop=30,saves=[10,30,70,100,-1])        
        
        # minimal_print(problem, final=problem.index == len(suite) - 1)"""

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html") 
