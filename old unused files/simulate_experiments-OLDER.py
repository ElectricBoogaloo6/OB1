#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 1-10-2020
# OB1 is a reading-model that simulates the processes behind reading in the brain.
# Here we simulate performance on two experimental word-recognition tasks:
# a flanker task and a sentence reading task

from __future__ import division
from collections import defaultdict
import re
from reading_common import stringToBigramsAndLocations, calcBigramExtInput, calcMonogramExtInput, get_stimulus_text_from_file, calc_word_attention_right
from reading_functions import my_print, get_threshold, getMidwordPositionForSurroundingWord, is_similar_word_length, \
    calc_saccade_error, norm_distribution, normalize_pred_values
from read_saccade_data import get_freq_pred_files, get_freq_and_syntax_pred
import numpy as np
import pickle
import parameters as pm
import sys
if pm.visualise:
    import Visualise_reading


def experiment_simulation_EEG(filename, parameters):

    lexicon = []
    lengtes = []
    all_data = []

    # load dictionaries (French Lexicon Project database) and generate list of individual words
    if pm.language == "french":
        word_freq_dict, word_pred_values = get_freq_pred_files_fr()
        # Replace prediction values with syntactic probabilities
        if pm.use_grammar_prob:
            print("grammar prob not implemented yet")
            raise NotImplemented
            #word_pred_values = get_freq_and_syntax_pred()["pred"]
        if pm.uniform_pred:
            print("Replacing pred values with .25")
            word_pred_values[:] = 0.25
    else:
        print("language not implemented yet")
        raise NotImplemented

    max_frequency_key = max(word_freq_dict, key=word_freq_dict.get)
    max_frequency = word_freq_dict[max_frequency_key]
    print("Length text: " + str(len(individual_words)) + "\nLength pred: " + str(len(word_pred_values)))
    word_pred_values = word_pred_values[0:len(individual_words)]

    # Make individual words dependent variables
    TOTAL_WORDS = len(individual_words)
    print("LENGTH of freq dict: "+str(len(word_freq_dict)))
    print("LENGTH of individual words: "+str(len(individual_words)))

    # make experiment lexicon (= dictionary + words in experiment)
    # make sure it contains no double words
    n_known_words = len(lexicon)  # MM: nr of words known to model
    for word in individual_words:
        if word not in lexicon:
            lexicon.append(word)

    # Make lexicon dependent variables
    LEXICON_SIZE = len(lexicon)

    # Normalize word inhibition to the size of the lexicon.
    lexicon_normalized_word_inhibition = (100.0/LEXICON_SIZE) * pm.word_inhibition

    # MM: list with trheshold values for words in lexicon
    for i, word in enumerate(lexicon):
        lexicon_thresholds_np[i] = get_threshold(word,
                                                 word_freq_dict,
                                                 max_frequency,
                                                 pm.wordfreq_p,
                                                 pm.max_threshold)
        lexicon_index_dict[word] = i
        lexicon_word_activity[word] = 0.0


    # lexicon bigram dict
    N_ngrams_lexicon = []  # GS list with amount of ngrams per word in lexicon
    for word in range(LEXICON_SIZE):
        lexicon[word] = " "+lexicon[word]+" "
        [all_word_bigrams, bigramLocations] = stringToBigramsAndLocations(lexicon[word])
        lexicon[word] = lexicon[word][1:(len(lexicon[word]) - 1)]  # to get rid of spaces again
        lexicon_word_bigrams[lexicon[word]] = all_word_bigrams
        N_ngrams_lexicon.append(len(all_word_bigrams) + len(lexicon[word]))  # GS append to list of N ngrams

    print("Amount of words in lexicon: ", LEXICON_SIZE)
    print("Amount of words in text:", TOTAL_WORDS)
    print("")

    # word-to-word inhibition matrix (redundant? we could also (re)compute it for every trial; only certain word combinations exist)

    print ("Setting up word-to-word inhibition grid...")
    # Set up the list of word inhibition pairs, with amount of bigram/monograms
    # overlaps for every pair. Initialize inhibition matrix with false.
    word_inhibition_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=bool)  # PK this matrix was not initialized
    word_overlap_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=int)

    complete_selective_word_inhibition = True
    overlap_list = {}

    for other_word in range(LEXICON_SIZE):
        for word in range(LEXICON_SIZE):
            # GS Take word length into account here instead of below, where act of lexicon words is determinied
            if not is_similar_word_length(lexicon[word], lexicon[other_word]) or lexicon[word] == lexicon[other_word]:
                continue
            else:
                bigrams_common = []
                bigrams_append = bigrams_common.append
                bigram_overlap_counter = 0
                for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
                    if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
                        bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
                        lexicon_word_bigrams_set[lexicon[word]] = set(lexicon_word_bigrams[lexicon[word]])
                        bigram_overlap_counter += 1

                monograms_common = []
                monograms_append = monograms_common.append
                monogram_overlap_counter = 0
                unique_word_letters = ''.join(set(lexicon[word]))

                for pos in range(len(unique_word_letters)):
                    monogram = unique_word_letters[pos]
                    if monogram in lexicon[other_word]:
                        monograms_append(monogram)
                        monogram_overlap_counter += 1

                # take into account both bigrams and monograms for inhibition counters (equally)
                total_overlap_counter = bigram_overlap_counter + monogram_overlap_counter

                # GS if word or other word is larger than the initial lexicon
                # (without PSC), overlap counter = 0, because words that are not
                # known should not inhibit
                if word >= n_known_words or other_word >= n_known_words:
                    total_overlap_counter = 0
                min_overlap = pm.min_overlap  # MM: currently 2

                if complete_selective_word_inhibition:
                    if total_overlap_counter > min_overlap:
                        word_overlap_matrix[word, other_word] = total_overlap_counter - min_overlap
                    else:
                        word_overlap_matrix[word, other_word] = 0
                else:  # is_similar_word_length
                    if total_overlap_counter > min_overlap:
                        word_inhibition_matrix[word, other_word] = True
                        word_inhibition_matrix[other_word, word] = True
                        overlap_list[word, other_word] = total_overlap_counter - min_overlap
                        overlap_list[other_word, word] = total_overlap_counter - min_overlap
                        sys.exit('Make sure to use slow version, fast/vectorized version not compatible')

    # Save overlap matrix, with individual words selected
    output_inhibition_matrix = 'Data/Inhibition_matrix_fr.dat'
    with open(output_inhibition_matrix, "wb") as f:
        pickle.dump(np.sum(word_overlap_matrix, axis=0)[individual_to_lexicon_indices], f)
    print("Inhibition grid ready.")
    print("")
    print("BEGIN EXPERIMENT")
    print("")

# Initialize Parameters
    regression = False
    wordskip = False
    refixation = False
    forward = False
    saccade_distance = 0  # Amount of characters
    fixation_duration = 0
    end_of_text = False  # Is set to true when end of text is reached.
    fixation = 0  # The iterator that indicates the element of fixation in the text
    # (this iterator can go backwards as well, with regressions).
    trial = 0
    fixation_counter = 0  # The iterator that increases +1 with every next fixation,
    # to expand all_data with every next fixation.

    # If eye position is to be in a position other than that of the word
    # middle, offset will be negative/positive (left/right) and will represent
    # the number of letters to the new position. It's value is reset before a
    # new saccade is performed.
    OffsetFromWordCenter = 0
    offset_previous = 0
    attendWidth = 4.0
    nextEyePosition = 0
    saccade_distance = 0
    saccade_error = 0
    refixation_type = 0
    wordskip_pass = 0
    saccade_type_by_error = 0
    attendposition_change = False
    attendposition_change_counter = 0
    width_change_delay = 0
    CYCLE_SIZE = 25  # milliseconds that one model cycle is supposed to last (brain time, not model time)
    allocated_dict = defaultdict(list)  # MM: dictionary that will contain allocated words
    # defaultdict = dict that creates new entry each time that key does not yet exist.
    # (list): new entry will be empty list
    salience_position_new = pm.salience_position
    previous_fixated_words = None
    previous_lexicon_values = None
    reset_pred_previous = False
    N_in_allocated = 0
    N1_in_allocated = 0
    to_pauze = False

    if pm.visualise:
        Visualise_reading

    all_data[trial] = {'foveal word': individual_words[fixation],
                                      'foveal word text index': fixation,
                                      'stimulus': [],
                                      'word activities per cycle': [],
                                      'fixation duration': 0,
                                      'recognized words indices': [],
                                      'attentional width': attendWidth,
                                      'exact recognized words positions': [],
                                      'eye position': 0,
                                      'refixated': refixation,
                                      'wordskipped': wordskip,
                                      'regressed': regression,
                                      'forward': forward,
                                      'fixation word activities': [],
                                      'word threshold': 0,
                                      'word frequency': 0,
                                      'word predictability': 0,
                                      'saccade error': saccade_error,
                                      'saccade distance': int(np.round(saccade_distance)),
                                      'wordskip pass': wordskip_pass,
                                      'refixation type': refixation_type,
                                      'saccade_type_by_error': saccade_type_by_error,
                                      'Offset': OffsetFromWordCenter,
                                      'relative landing position': offset_previous}

    # generate / read in stimuli list from file (fixed items for both experiments)
    import pandas as pd
    if pm.use_sentence_task:
        stim = pd.read_table('E:/Projects/2020_reading/SentenceReading/Stimuli_all_csv.csv', sep=',')
    elif pm.use_flanker_task:
        stim = pd.read_table('E:/Projects/2020_reading/Flanker/Stimuli_all_csv.csv', sep=',')
    lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity

    my_print('attendWidth: '+str(attendWidth))

    # BEGIN EXPERIMENT
    # loop over trials?
    for trial in range(0,len(stimuli)):

        stimulus = stim['all'][trial]

        individual_words = []
        lengtes=[]
        textsplitbyspace = stimulus.split(" ")

        for word in textsplitbyspace:
            if word.strip() != "":
                new_word = np.unicode_(word.strip()) #For Python2
            individual_words.append(new_word)
            lengtes.append(len(word))

        word_thresh_dict = {}
        # for each word, compute threshold based on freq and pred
        for word in individual_words:
            word_thresh_dict[word] = get_threshold(word,
                                                   word_freq_dict,
                                                   max_frequency,
                                                   pm.wordfreq_p,
                                                   pm.max_threshold)
            try:
                word_freq_dict[word]
            except KeyError:
                word_freq_dict[word] = 0

        # force fixation in center of all words on screen (1-5 words can appear on screen)
        fixation_counter=0
        all_data[trial]['stimulus'] = stimulus

        for word in range(len(stimulus.split(" "))):
            # "Word activities per cycle" is a dict containing the stimulus' words.
            # For every word there is a list that will keep track of the activity per cycle.
            all_data[trial]['word activities per cycle'].append(
                {stimulus.split(" ")[word+1]: []}
            )

        # Adjust lexicon thresholds with predictability values,
        # only when words in stimulus
        # MM: why done here and not at top in one go for whole txt?

        norm_pred_values = normalize_pred_values(pm.wordpred_p, word_pred_values\
                                                 [fix_start:fix_end])
        previous_fixated_words = lexicon_fixated_words
        previous_lexicon_values = lexicon_thresholds_np[lexicon_fixated_words]
        reset_pred_previous = True
        lexicon_thresholds_np[lexicon_fixated_words] = lexicon_thresholds_np\
                                                       [lexicon_fixated_words] * norm_pred_values

        # get allNgrams for current trial
        [allNgrams, bigramsToLocations] = stringToBigramsAndLocations(stimulus)
        allMonograms = []
        allBigrams = []

        for ngram in allNgrams:
            if len(ngram) == 2:
                allBigrams.append(ngram)
            else:
                allMonograms.append(ngram)
        allBigrams_set = set(allBigrams)
        allMonograms_set = set(allMonograms)

        # enter the cycle-loop that builds word activity with every cycle
        my_print("fixation: " + individual_words[fixation])

        amount_of_cycles = 0
        amount_of_cycles_since_attention_shifted = 0
            ### stimulus on screen for 150 ms (flanker) or 200 ms (sentence)
        if pm.use_sentence_task:
            ncycles = 8
        if pm.use_flanker_task:
            ncycles = 6

        while amount_of_cycles_since_attention_shifted < ncycles:

            unitActivations = {}  # reset after each trials
            lexicon_activewords = []
            # Only the words in "lexicon_activewords" will later participate in word-to-word inhibition.
            # As such, less word overlap pairs will be called when calculating inhibition,
            # so to speed up the code.
            # Stores the indexes of the words in the lexicon are stored.

            # Reset
            word_input = []
            word_input_np.fill(0.0)
            lexicon_word_inhibition_np.fill(0.0)
            lexicon_word_inhibition_np2.fill(0.0)
            lexicon_activewords_np.fill(False)

            crt_fixation_word_activities = [0, 0, 0, 0, 0]
            ### Calculate ngram activity

            ### activation of word nodes
                # taking nr of ngrams, word-to-word inhibition etc. into account


            ### determine target word (= only word on screen, or word in center)

            ### save activation for target word  for every cycle

            ### "evaluate" response
                ## e.g. through the Bayesian model Martijn mentioned (forgot to write it down),
                ## or some hazard function that expresses the probability
                ## of the one-choice decision process terminating in the
                ## next instant of time, given that it has survived to that time?
            ### if target word has been recognized (e.g. above threshold in time):
                ### response = word
                ### RT = moment in cylce
            ### if target word has not been recognized:
                ### response = nonword
                ### RT = moment in cycle

            print("end of trial")



def sentencereading_simulation(filename,parameters):

    # load dictionaries (French Lexicon Project database) and generate list of individual words
        ### language = "french"
        ### Compute word_freq + word_pred for thresholds

    # make experiment lexicon (= dictionary + words in experiment?)
        ### Normalize word inhibition to the size of the lexicon

    # for each word, compute threshold based on freq and pred

    # lexicon bigram dict

    # word-to-word inhibition matrix (redundant? we could also (re)compute it for every trial; only certain word combinations exist)

    # generate / read in stimuli list from file (fixed items for both experiments)

    # BEGIN EXPERIMENT
    # loop over trials?
    for trial in range(0,len(stimuli)):

        # force fixation in center of all words on screen (1-5 words can appear on screen)

        # get allNgrams for current trial

        # enter the cycle-loop that builds word activity with every cycle
            ### stimulus on screen for 150 ms (flanker) or 200 ms (sentence)

            ### Calculate ngram activity

            ### activation of word nodes
                # taking nr of ngrams, word-to-word inhibition etc. into account

            ### Read cue location from file!
            ### determine target word

            ### save activation for target word for every cycle

            ### "evaluate" response
                ### if target word has been recognized (e.g. above threshold before 200 ms - 8 cycles?):
                ### response = correct
            ### if target word has not been recognized:
                ### response = incorrect

            print("end of trial")
