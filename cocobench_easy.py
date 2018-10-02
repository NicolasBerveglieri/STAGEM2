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

	
### input
suite_name = "bbob"  #-biobj"
output_folder = "folder_test"
fmin = scipy.optimize.fmin
budget_multiplier = 1  # increase to 10, 100, ...

### prepare
suite = cocoex.Suite(suite_name, "", "")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()
data_directory = "mono_EGO"+"_data/"
print("\n\n\n\n\nwadup\n\n\n\n\n")



### go
cpt = 0 
f = 1
for problem in suite:  # this loop will take several minutes or longer



    if cpt % 15 == 0:
        print(cpt)
        print(problem.name)
        test.surface_test_mono(problem,"function_"+str(f))
        sys.exit()
        f+=1
    cpt+=1
    if f == 25:
        sys.exit()
"""    

    direc = data_directory+"problem_1_"+problem.name
    #print("taille du probleme: ",problem.number_of_variables)
    print(problem.upper_bounds)
    ego_mono.EGO_mono(problem,direc,solution_size=problem.number_of_variables)
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    x0 = problem.initial_solution
    sys.exit()
    if test == 0:
    	print("initial x0 value:",x0)
    # apply restarts while neither the problem is solved nor the budget is exhausted
    whilesize = 0
    while (problem.evaluations < problem.dimension * budget_multiplier
           and not problem.final_target_hit):
        whilesize+=1
	print(whilesize)
	if test == 0:
		print("avant fmin:",x0,problem.lower_bounds)
        a = fmin(problem, x0, disp=False)  # here we assume that `fmin` evaluates the final/returned solution
	if test == 0:
		print("apres fmin",x0,problem.lower_bounds,problem.best_observed_fvalue1)
        x0 = problem.lower_bounds + ((rand(problem.dimension) + rand(problem.dimension)) *
                    (problem.upper_bounds - problem.lower_bounds) / 2)
	if test == 0:
		print("apres 2nd",x0)
	print("final target hit??", problem.final_target_hit,problem.evaluations,problem.dimension, budget_multiplier,problem.dimension* budget_multiplier)
		
    minimal_print(problem, final=problem.index == len(suite) - 1)
"""

"""
### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
""" 
