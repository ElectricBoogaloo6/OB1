# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from parameters import return_params

pm=return_params()

from reading_simulation import reading_simulation
from get_scores import get_scores


def reading_function(parameters_rf, names): #TODO: move this. Probably should go to reading_functions
    global OLD_DISTANCE
    global N_RUNS
    filename = "PSC_ALL"
    filepath_psc = "PSC/" + filename + ".txt"

### For testing (loading past results instead of running simulation)
#    with open("Results/all_data.pkl","r") as f:
#        all_data = pickle.load(f)
#    with open("Results/unrecognized.pkl","r") as f:
#        unrecognized_words = pickle.load(f)
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

