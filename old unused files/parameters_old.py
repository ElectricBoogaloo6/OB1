# CHANGED
# -*- coding: UTF-8 -*-
import time

__author__ = 'Sam van Leipsig, Phillip Kersten'

use_grammar_prob = False # True for using grammar probabilities, False for using cloze, overwritten by uniform_pred
uniform_pred = False  # Overwrites cloze/grammar probabilities with 0.25 for all words

include_sacc_type_sse = True  # Include the sse score based on the saccade type probability plot
sacc_type_objective = "total"  # If "total" all subplots will be included in the final sse,
                               #  single objectives can be "length", "freq" or "pred"

include_sacc_dist_sse = True  # Include the SSE score derived from the saccade_distance.png plot

tuning_measure = "SSE"  # can be "KL" or "SSE"
discretization = "bin"  # can be "bin" or "kde"
objective = []  # empty list for total SSE/KL, for single objectives: "total viewing time",
                # "Gaze durations", "Single fixations", "First fixation duration",
                # "Second fixation duration", "Regression"

output_dir = time.time()
epsilon = 0.1  # Step-size for approximation of the gradient


#print("_----PARAMETERS----_")
#print("reading in " + language)
#if use_grammar_prob:
#    print("Using syntax probabilities")
#else:
#    print("Using cloze probabilities")
#if optimize:
#    print("Using: "+tuning_measure)
#    if any(objective):
#        print("Single Objective: "+tuning_measure+" of "+objective)
#    else:
#        print("Using total "+tuning_measure)
#    print("Step-size: "+str(epsilon))
#print("-------------------")


## Monoweight = 1
decay = -0.08 #-0.053
bigram_to_word_excitation = 3.09269333333 #2.18 # inp. divded by #ngrams, so this param estimates excit per word [diff from paper]
bigram_to_word_inhibition = -0.20625 #-0.6583500000000001 # -0.55
word_inhibition = -0.0165 #-0.016093 #-0.011 # -0.002

letPerDeg = .3
min_activity = 0.0
max_activity = 1.3

## Attentional width
max_attend_width = 5.0
min_attend_width = 3.0
attention_skew = 4  # 1 equals symmetrical distribution # 4 (paper)
bigram_gap = 3  # How many in btw letters still lead to bigram? 6 (optimal) # 3 (paper)
min_overlap = 2
refix_size = 0.2
salience_position = 4.99  # 1.29 # 5 (optimal) # 1.29 (paper)
corpora_repeats = 0 # how many times should corpus be repeated? (simulates diff. subjects)


## Model settings
frequency_flag = True # use word freq in threshold
prediction_flag = True
similarity_based_recognition = True
use_saccade_error = True
use_attendposition_change = True # attend width influenced by predictability next wrd
visualise = False
slow_word_activity = False
print_all = False
pauze_allocation_errors = False
use_boundary_task = False

## Saccade error
sacc_optimal_distance = 9.99  # 3.1 # 7.0 # 8.0 (optimal) # 7.0 (paper)
saccErr_scaler = 0.2  # to determine avg error for distance difference
saccErr_sigma = 0.17 # basic sigma
saccErr_sigma_scaler = 0.06 # effect of distance on sigma

## Fixation duration# s
mu, sigma = 10.09, 5.36  # 4.9, 2.2 # 5.46258 (optimal), 4 # 4.9, 2.2 (paper)
distribution_param = 5.0  #1.1

## Threshold parameters
max_threshold = 1
wordfreq_p = 0.4 # Max prop decrease in thresh. for highest-freq wrd [different definition than in papers]
wordpred_p = 0.4 # Currently not used

## Threshold parameters
linear = False

use_sentence_task = True
use_flanker_task = False