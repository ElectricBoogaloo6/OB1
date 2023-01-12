# -*- coding: utf-8 -*-
# 1-10-2020 Noor Seijdel
# In this file, "simulate_experiments" is called and the results are stored 

from simulate_experiments import simulate_experiments
from analyse_data_pandas import get_results, get_results_simulation
import pickle
import time
import numpy as np
import parameters_exp as pm
import pandas as pd
from get_parameters import get_params

# Get parameters for tuning
parameters, bounds, names = get_params(pm)

# Init distance variables for the reading function used in tuning
OLD_DISTANCE = np.inf
N_RUNS = 0

# Now name output files. Fst default, then the ones suitable for particular sim.
output_file_all_data, output_file_unrecognized_words = ("Results/all_data"+pm.language+".pkl","Results/unrecognized"+pm.language+".pkl")
if(pm.use_sentence_task):
    output_file_all_data, output_file_unrecognized_words = ("Results/alldata_Sentence.pkl","Results/unrecognized_Sentence.pkl")
if(pm.use_flanker_task):
    output_file_all_data, output_file_unrecognized_words = ("Results/alldata_Flanker.pkl","Results/unrecognized_Flanker.pkl")
start_time = time.time()

if pm.run_exp:
    # Run the reading model
    (lexicon, all_data, unrecognized_words) = simulate_experiments(task,parameters=[])
    # Save results: all_data...
    all_data_file = open(output_file_all_data,"wb")
    df_alldata = pd.DataFrame(all_data)
    df_alldata.to_pickle(output_file_all_data)
    #pickle.dump(all_data, all_data_file)
    all_data_file.close()
    	# ...and unrecognized words
    unrecognized_file = open(output_file_unrecognized_words, "wb")
    pickle.dump(unrecognized_words, unrecognized_file)
    unrecognized_file.close()
    
    with open("unrecognized.txt", "w") as f:
                f.write("Total unrecognized: " + str(len(unrecognized_words)))
                f.write("\n")
                for uword in unrecognized_words:
                        f.write(str(uword))
                f.write("\n")
    
    with open("alldata.txt", "w") as f:
                f.write("\n")
                for uword in all_data:
                        f.write(str(uword))
                f.write("\n")


if pm.analyze_results:
	get_results_simulation(output_file_all_data,output_file_unrecognized_words)

time_elapsed = time.time()-start_time
print("Time elapsed: "+str(time_elapsed))
