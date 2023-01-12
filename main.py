# -*- coding: utf-8 -*-
# 1-10-2020 Noor Seijdel
# In this file, "simulate_experiments" is called and the results are stored

# NV: Merged main and main_exp into this file.
# Specify what tasks to run in parameters.
# If any of the experimental tasks are chosen, run body of what was previosuly main_exp,
# of PSC is chosen, run body of what used to be main.py

from datetime import datetime
import logging
import scipy
import numpy as np
import time
import pickle

from simulate_experiments import simulate_experiments
from get_parameters import get_params  
from analyse_data_pandas import get_results_simulation
from parameters import return_params
from reading_simulation import reading_simulation
from reading_function_optimize import reading_function
from analyse_data_pandas import get_results

now = datetime.now()
dt_string = now.strftime("_%d_%m_%Y_%H-%M-%S")

# will create a new file everytime, stamped with date and time. #TODO; build system to keep only last X logs
logging.basicConfig(filename=f'logs/logfile{dt_string}.log', encoding='utf-8', force=True,
                    filemode='w', level=logging.DEBUG, format='%(name)s %(levelname)s:%(message)s')

logger = logging.getLogger(__name__)

pm = return_params()  # NV: get all parameters as an object
task = pm.task_to_run  # NV: get name of task as individual object

logger.debug(pm)

print("Task:"+task)
print("_----PARAMETERS----_")
print("reading in " + pm.language)

# NV: added uniform pred, which overwrites the 2 others if set to true

#TODO: move to consistency checks in simulate_expriments
if pm.uniform_pred:
    print("Using uniform 0.25 probabilities")
elif pm.use_grammar_prob:
    print("Using syntax probabilities")
else:
    print("Using cloze probabilities")
    
if pm.optimize:
    print("Using: "+pm.tuning_measure)
    if any(pm.objective):
        print("Single Objective: "+pm.tuning_measure+" of "+pm.objective)
    else:
        print("Using total "+pm.tuning_measure)
    print("Step-size: "+str(pm.epsilon))
print("-------------------")

output_file_all_data, output_file_unrecognized_words = (
    "Results/alldata_"+task+".pkl", "Results/unrecognized_"+task+".pkl")

start_time = time.perf_counter()

if pm.is_experiment:  # NV: if the task is an experiment
    if pm.run_exp:
        # Run experiment simulation
        (lexicon, all_data, unrecognized_words) = simulate_experiments(task, pm)
        # Save results: all_data
        # NV: Changed syntax of writing to file.
        with open(output_file_all_data, "wb") as all_data_file:
            pickle.dump(all_data, all_data_file)
        # ...and unrecognized words
        with open(output_file_unrecognized_words, "wb") as unrecognized_file:
            pickle.dump(unrecognized_words, unrecognized_file)

        # NV: write unrecognized and recognized words also to text file.
        # For consulting manually?
        with open("Results/unrecognized.txt", "w") as f:
            f.write("Total unrecognized: " + str(len(unrecognized_words)))
            f.write("\n")
            for uword in unrecognized_words:
                f.write(uword)
            f.write("\n")

        with open("Results/alldata.txt", "w") as f:
            f.write("\n")
            for uword in all_data:
                f.write(str(uword))
            f.write("\n")

    # TODO: look into this
    if pm.analyze_results:  # NV: what does this do exactly?
        get_results_simulation(task, output_file_all_data,
                               output_file_unrecognized_words)

    if pm.optimize:  # NV: not coded for experiments yet (check copies)
        pass

else:  # NV: if not a task, run simulation of text reading (german, PSC)


    if pm.language == "german":
        filename = "PSCmini"  # "PSC_ALL"
        filepath_psc = "PSC/" + filename + ".txt"
    # NV: for all other languages
    else:
        raise NotImplementedError("language is not implemented for text reading")

    if pm.run_exp:
        # Run the reading model
        (lexicon, all_data, unrecognized_words, highest_act_words,
         act_above_threshold,
         read_words) = reading_simulation(filepath_psc, parameters=pm)
        # GS these can be used for some debugging checks
        #highest_act_words, act_above_threshold, read_words
        # Save results: all_data...
        with open(output_file_all_data, "wb") as f:
            pickle.dump(all_data, f)
        # ...and unrecognized words
        with open(unrecognized_file, 'wb') as f:
            pickle.dump(output_file_unrecognized_words, f)

    if pm.analyze_results:
        get_results(filepath_psc, output_file_all_data,
                    output_file_unrecognized_words)

    if pm.optimize:
        epsilon = pm.epsilon
        parameters, bounds, names = get_params(pm)
        OLD_DISTANCE = np.inf
        N_RUNS = 0
        results = scipy.optimize.fmin_l_bfgs_b(func=reading_function,
                                               args=(names),
                                               x0=np.array(parameters),
                                               bounds=bounds,
                                               approx_grad=True, disp=True,
                                               epsilon=epsilon)
        # TODO: look into this. Doesnt work just yet
        with open("results_optimization.pkl", "wb") as f:
            pickle.dump(results, f)


time_elapsed = time.perf_counter()-start_time
print("Time elapsed: "+str(time_elapsed))

