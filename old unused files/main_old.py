# -*- coding: utf-8 -*-
# CHANGED
from reading_simulation import reading_simulation
from analyse_data_pandas import get_results
import multiprocessing as mp
import pickle
import cProfile
import pstats
from analyse_data_pandas import get_results
import pickle
import scipy
import time
import numpy as np
from get_scores import get_scores
import parameters as pm
import pandas as pd
from get_parameters import get_params

# Get parameters for tuning
parameters, bounds, names = get_params(pm)

# Init distance variables for the reading function used in tuning
OLD_DISTANCE = np.inf
N_RUNS = 0


# Reading function for tuning (called by scipy's L-BFGS-B optimizing function)
def reading_function(parameters_rf):
	global OLD_DISTANCE
	global N_RUNS
	filename = "PSC_ALL"
	filepath_psc = "PSC/" + filename + ".txt"

### For testing (loading past results instead of running simulation)
#	with open("Results/all_data.pkl","r") as f:
#		all_data = pickle.load(f)
#	with open("Results/unrecognized.pkl","r") as f:
#		unrecognized_words = pickle.load(f)
###
	# Run the simulation

	(lexicon, all_data, unrecognized_words) = reading_simulation(filepath_psc, parameters_rf)
	# Evaluate run and retrieve error-metric
	distance = get_scores(filename, all_data, unrecognized_words)

        # Save parameters when distance is better than previous
	write_out = pd.DataFrame(np.array([names, parameters_rf]).T)
	if distance < OLD_DISTANCE:
        	write_out.to_csv(str(distance)+"_"+pm.tuning_measure+"parameters.txt", index=False, header=["name", "value"])
        	OLD_DISTANCE = distance

	p = ""

	for param, name in zip(parameters_rf, names):
		p += name +": "
		p += str(param)
		p += "\n"

	# Save distances for plotting convergence
	with open("dist.txt", "a") as f:
		f.write("run "+str(N_RUNS)+": "+str(int(distance))+"\n")
		f.write(p)
		f.write("\n")
	N_RUNS += 1 
	return distance

if pm.language == "german":
	filename = "PSCmini" #"PSC_ALL"
	filepath_psc = "PSC/" + filename + ".txt"
# The reading model reads dutch but there is no data to compare it to yet
if pm.language == "dutch":
	raise NotImplementedError
	filename = "PSC/words_dutch.pkl"

output_file_all_data, output_file_unrecognized_words = ("Results/all_data"+pm.language+".pkl","Results/unrecognized"+pm.language+".pkl")
start_time = time.time()

if pm.run_exp:
	# Run the reading model
	(lexicon, all_data, unrecognized_words,highest_act_words, act_above_threshold, read_words) = reading_simulation(filepath_psc, parameters=[])
    # GS these can be used for some debugging checks
    #highest_act_words, act_above_threshold, read_words
	# Save results: all_data...
	all_data_file = open(output_file_all_data,"wb")
	pickle.dump(all_data, all_data_file)
	all_data_file.close()
	# ...and unrecognized words
	unrecognized_file = open(output_file_unrecognized_words, "wb")
	pickle.dump(unrecognized_words, unrecognized_file)
	unrecognized_file.close()

if pm.analyze_results:
	get_results(filepath_psc,output_file_all_data,output_file_unrecognized_words)

#if pm.optimize:
#	epsilon = pm.epsilon
#	results = scipy.optimize.fmin_l_bfgs_b(func=reading_function, x0=np.array(parameters), bounds=bounds, approx_grad=True, disp=True , epsilon=epsilon)
#	with open("results_optimization.pkl", "wb") as f:
#		pickle.dump(results, f)

time_elapsed = time.time()-start_time

print('unrecognized N: ',len(unrecognized_words))

print("Time elapsed: "+str(time_elapsed))
