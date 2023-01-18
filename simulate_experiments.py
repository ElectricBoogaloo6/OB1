#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 1-10-2020 Noor Seijdel
# OB1 is a reading-model that simulates the processes behind reading in the brain.
# Here we simulate performance on experimental word-recognition tasks:
# a flanker task, a sentence reading task and an embeddedwords task

#NOTE: the code is united, and now (is supposed to) run both experiments and normal text reading in german (PSC)

from __future__ import division
import logging
import numpy as np
import sys
import pandas as pd
import pickle
import os

from parameters import return_params
from freq_pred_files import get_freq_files, get_pred_files
from affixes import get_suffix_file  # , get_prefix_file
from reading_functions import get_threshold, is_similar_word_length, extract_stem, word_stem_match
from reading_common import stringToBigramsAndLocations, calcBigramExtInput, calcMonogramExtInput
from analyse_data_plot import plot_runtime, plot_inhib_spectrum

logger = logging.getLogger(__name__)


def simulate_experiments(task, pm):

    # NV: consistency checks

    if pm.affix_system and not pm.affix_implemented:

        print('language/task does not support affix system yet! Switching to non-affix version')
        logger.warn('language/task does not support affix system yet! Switching to non-affix version')

        affixes = []

        pm.affix_system = False

    if pm.use_grammar_prob and not pm.POS_implemented:

        print("language or task has no POS data. \
             Unable to use grammar pred values for this task. Switching to unfiform pred values")
        logger.warn("language or task has no POS data. \
                Unable to use grammar pred values for this task. Switching to unfiform pred values")

        pm.uniform_pred = True
        pm.use_grammar_prob = False

    if pm.trial_ends_on_key_press and task in ('Sentence', 'Classification'):
        print("Task is not a classical word recognition task and cannot be terminated by key press. \
              Switching to trial_ends_on_key_press = False")
        logger.warn("Task is not a classical word recognition task and cannot be terminated by key press. \
                     Switching to trial_ends_on_key_press = False")

        pm.trial_ends_on_key_press = False

    # NV: conditional import, so has to be imported after pm is specified
    if pm.visualise:
        import Visualise_reading

    lexicon = []
    lengtes = []
    all_data = []

    # NV: generate list of individual words and their lengths from stimulus file
    individual_words = []
    lengtes = []

    #KM: stopping 'prime' and 'all' being added to the NONWORDS 
    # German
    #file_nonwords = pd.read_csv(r"C:\Users\Konstantin\Documents\OB1-1\Stimuli\EmbeddedWords_Nonwordslower_german_all_csv.csv", sep = ';')
    # French
    file_nonwords = pd.read_csv(r"C:\Users\Konstantin\Documents\OB1-1\Stimuli\EmbeddedWords_Nonwords_french_all_csv.csv", sep = ';')
    to_mask = file_nonwords[file_nonwords['condition'].str.contains('stem|suffix')]
    nonwords = list(to_mask['all'].str.split(' ', expand=True).stack().unique())
        #ORIGINAL
    textsplitbyspace = list(pm.stim['all'].str.split(
        ' ', expand=True).stack().unique())  # get stimulus words of task
    
    #NV: also add prime words to list if applicable
    if pm.is_priming_task:# and task != 'EmbeddedWords_German':
        textsplitbyspace.extend(list(pm.stim['prime'].str.split(
            ' ', expand=True).stack().unique()))

    # NV: generate individual_words array
    # KM avoiding adding NONWORDS
    for word in textsplitbyspace:
        if word.strip() != "" and word not in nonwords:
            # NV: add _ to begin and end of words, for affix recognition system
            new_word = f"_{word.strip().lower()}_"
            individual_words.append(new_word)
            lengtes.append(len(word))
    logger.debug(f'individual words: {individual_words}')

    # NV load appropriate dictionary
    # get file of words of task and their frequency and 200 most common words of language
    word_freq_dict_temp = get_freq_files(task, pm)

    # NV: also add _ to word_freq_dict, for affix modelling purposes.
    word_freq_dict = {}
    for word in word_freq_dict_temp.keys():
        word_freq_dict[f"_{word}_"] = word_freq_dict_temp[word]

    logger.debug('word freq dict (first 20): \n' +
                 str({k: word_freq_dict[k] for k in list(word_freq_dict)[:20]}))

    # if using affix system, import freq & pred data of afixes as well
    if pm.affix_system:

        suffix_freq_dict_temp = get_suffix_file(pm)
        suffix_freq_dict = {}
        for word in suffix_freq_dict_temp.keys():
            suffix_freq_dict[f"{word}_"] = suffix_freq_dict_temp[word]
        suffixes = list(suffix_freq_dict.keys())

        # NV: NOTE: at the moment, only suffixes are implemented. To implement prefixes as well, head to read_saccade_data and affixes.py
        prefixes = []
        prefix_freq_dict = {}

        affix_freq_dict = suffix_freq_dict | prefix_freq_dict  # NV: merge 2 dictionnaries
        affixes = prefixes+suffixes

        logger.debug(affixes)

        # NV: add affix freq and pred to list
        # affixes have no pred values, so fill in with mean pred of normal words
        word_freq_dict = word_freq_dict | affix_freq_dict

        logger.debug('word freq dict (with affixes): ' + str(word_freq_dict))

    if pm.use_grammar_prob:

        # FIXME:  what is the use of this code block? make pred files? TODO: figure out. Might dissolve confusion
        
        word_pred_values = get_pred_files(task, pm)
        
        # Overwrite with 0.25
        if pm.uniform_pred:
            print("Replacing pred values with .25")
            word_pred_values[:] = 0.25

        grammar_weight = pm.grammar_weight

        POS_corpus = pd.read_table('Texts/' + pm.language +
                                   '_lexicon_tagged_2000.csv', sep=',', encoding='utf-8')
        POS_task = pd.read_table('./Stimuli/' + task +
                                 '_task_POS_csv.csv', sep=',', encoding='utf-8')

        # Create dictionary with all the words in the lexicon and their POS
        POS_keys = POS_corpus['Words'].to_list() + POS_task['Words'].to_list()
        POS_values = POS_corpus['POS'].to_list() + POS_task['POS'].to_list()
        POSdict = dict(zip(POS_keys, POS_values))

        # Create the dictionary with the POS probabilities for the word before and after the target
        Grammar_keys = POS_task['POSbef'].to_list()
        Grammar_valuesb = POS_task['Probbef'].to_list()
        Grammar_valuesa = POS_task['Probaft'].to_list()
        Grammardict = dict(zip(Grammar_keys, [Grammar_valuesb, Grammar_valuesa]))


    max_frequency_key = max(word_freq_dict, key=word_freq_dict.get)
    max_frequency = word_freq_dict[max_frequency_key]
    print("max freq:" + str(max_frequency))
    print("Length text: " + str(len(individual_words)))
    logger.info("max freq:" + str(max_frequency))
    logger.info("Length text: " + str(len(individual_words)))

    # Make individual words dependent variables
    TOTAL_WORDS = len(individual_words)
    print("LENGTH of freq dict: "+str(len(word_freq_dict)))
    print("LENGTH of individual words: "+str(len(individual_words)))
    logger.info("LENGTH of freq dict: "+str(len(word_freq_dict)))
    logger.info("LENGTH of individual words: "+str(len(individual_words)))

    # make experiment lexicon (= dictionary + words in experiment)
    for word in individual_words:  # make sure it contains no double words
        if word not in lexicon:
            lexicon.append(word)

    if(len(word_freq_dict) > 0):
        for freq_word in word_freq_dict.keys():
            if freq_word.lower() not in lexicon:
                # NV:word_freq_dict already contains all target words of task +eventual flankers or primers, determined in create_freq_pred_files. So the first part of individual words is probably double work
                # ANSWER: Actually, the word_freq_dict is only made for words, for which there is a frequency. Other words are discarded. So concatenating word_freq_dict with individual_words puts those words back! So here again, the important question is: Why the threshols in create_freq_pred_files?
                lexicon.append(freq_word.lower())

    lexicon_file_name = 'Data/Lexicon_'+task+'.dat'
    with open(lexicon_file_name, "wb") as f:
        pickle.dump(lexicon, f)

    n_known_words = len(lexicon)  # nr of words known to model

    logger.debug(f'size lexicon: {len(lexicon)}')

    # Make lexicon dependent variables
    LEXICON_SIZE = len(lexicon)

    logger.info("Amount of words in lexicon: " + str(LEXICON_SIZE))
    logger.info("Amount of words in text:" + str(TOTAL_WORDS))

    # Normalize word inhibition to the size of the lexicon.
    lexicon_normalized_word_inhibition = (
        100.0/LEXICON_SIZE) * pm.word_inhibition

    # Set activation of all words in lexicon to zero and make bigrams for each word.
    lexicon_word_activity = {}
    lexicon_word_bigrams = {}
    lexicon_word_bigrams_set = {}
    lexicon_index_dict = {}

    # Lexicon word measures
    lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_word_inhibition_np2 = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_activewords_np = np.zeros((LEXICON_SIZE), dtype=int)
    word_input_np = np.zeros((LEXICON_SIZE), dtype=float)
    lexicon_thresholds_np = np.zeros((LEXICON_SIZE), dtype=float)

    word_thresh_dict = {}

    # NV: find mimimum dict value of freq dict'
    # NV: instead of 0, insert 7th smallets value of dict (just to test)
    value_list = np.sort(list(word_freq_dict.values()))
    value_to_insert = value_list[7]

    # for each word, compute threshold based on freq and pred
    # MM: dit is eigenlijk raar, threshold voor lex en voor indivwrds
    # NV: Mee eens. Daarbij zitten alle individual_words in het lexicon, dus is dit dubbel werk toch? #TODO
    for word in individual_words:
        word_thresh_dict[word] = get_threshold(word,
                                               word_freq_dict,
                                               max_frequency,
                                               pm.wordfreq_p,
                                               pm.max_threshold)
        try:
            word_freq_dict[word]
        except KeyError:
            word_freq_dict[word] = value_to_insert

    # list with trheshold values for words in lexicon
    for i, word in enumerate(lexicon):
        lexicon_thresholds_np[i] = get_threshold(word,
                                                 word_freq_dict,
                                                 max_frequency,
                                                 pm.wordfreq_p,
                                                 pm.max_threshold)

        lexicon_index_dict[word] = i
        lexicon_word_activity[word] = 0.0

    # lexicon indices for each word of text (individual_words)
    individual_to_lexicon_indices = np.zeros((len(individual_words)), dtype=int)
    for i, word in enumerate(individual_words):
        individual_to_lexicon_indices[i] = lexicon.index(word)

    # NV: this code body is for stringToBigramsAndLocations execution, which splits words into their bigrams and positions within the word.
    # the first lines are for preparing the word for entering the stringToBigramsAndLocations function
    # TODO: in the long run, should change stringToBigramsAndLocations instead of changing its input
    N_ngrams_lexicon = []  # list with amount of ngrams per word in lexicon
    for word in lexicon:

        # NV: create local variable to modify without interfering
        # NV: strings are immutable so no need to deep copy
        word_local = word

        # NV:  determine if the word is an affix, and remove _'s
        is_suffix = False
        is_prefix = False
        # NV: if word is normal word, remove both _'s
        if word_local.startswith('_') and word_local.endswith('_'):
            word_local = word_local[1:-1]
        # NV: if prefix, remove first _ and set is_prefix to True
        elif word_local.startswith('_'):
            word_local = word_local[1:]
            is_prefix = True
        # idem for suffix
        elif word_local.endswith('_'):
            word_local = word_local[:-1]
            is_suffix = True

        else:
            raise SyntaxError("word does not start or stop with _ . Verify lexicon")

        # add spaces to word, important for function hereunder
        word_local = " "+word_local+" "
        # NV: convert words into bigrams and their locations
        (all_word_bigrams,
         bigramLocations) = stringToBigramsAndLocations(word_local, is_prefix, is_suffix)

        # append to list of N ngrams
        lexicon_word_bigrams[word] = all_word_bigrams
        # bigrams and monograms total amount
        N_ngrams_lexicon.append(len(all_word_bigrams) + len(word.strip('_')))

    print("Setting up word-to-word inhibition grid...")
    logger.info("Setting up word-to-word inhibition grid...")

    # Set up the list of word inhibition pairs, with amount of bigram/monograms overlaps for every pair. Initialize inhibition matrix with false.
    # NV: COMMENT: here is actually built an overlap matrix rather than an inhibition matrix, containing how many bigrams of overlap any 2 words have
    word_inhibition_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=bool)
    word_overlap_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE), dtype=int)

    complete_selective_word_inhibition = True  # NV: what does this do exactly? Move to parameters.py?
    overlap_list = {}

    # NV: first, try to fetch parameters of previous inhib matrix
    try:
        with open('Data/Inhib_matrix_params_latest_run.dat', "rb") as f:
            parameters_previous = pickle.load(f)

        size_of_file = os.path.getsize('Data/Inhibition_matrix_previous.dat')

        # NV: compare the previous params with the actual ones.
        # he idea is that the matrix is fully dependent on these parameters alone.
        # So, if the parameters are the same, the matrix should be the same.
        # The file size is also added as a check . Note: Could possibly be more elegant
        if str(lexicon_word_bigrams)+str(LEXICON_SIZE)+str(pm.min_overlap) +\
           str(complete_selective_word_inhibition)+str(n_known_words)+str(pm.affix_system) +\
           str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file) \
           == parameters_previous:

            previous_matrix_usable = True  # FIXME: turn off if need to work on inihibition matrix specifically

        else:
            previous_matrix_usable = False
    except:
        logger.info('no previous inhibition matrix')
        previous_matrix_usable = False

    # NV: if the current parameters correspond exactly to the fetched params of the previous run, use that matrix
    if previous_matrix_usable:
        with open('Data/Inhibition_matrix_previous.dat', "rb") as f:
            word_overlap_matrix = pickle.load(f)
        print('using pickled inhibition matrix')
        logger.info('\n using pickled inhibition matrix \n')

    # NV: else, build it
    else:
        print('building inhibition matrix')
        logger.info('\n Building new inhibition matrix \n')

        #overlap_percentage_matrix = np.zeros((LEXICON_SIZE, LEXICON_SIZE))
        complex_stem_pairs = []

        for other_word in range(LEXICON_SIZE):

            # as loop is symmetric, only go through every pair (word1-word2 or word2-word1) once.
            for word in range(other_word, LEXICON_SIZE):

                ### NV: 1. calculate monogram and bigram overlap
                
                # NV: bypass to investigate the effects of word-length-independent inhibition
                # if not is_similar_word_length(lexicon[word], lexicon[other_word]) or lexicon[word] == lexicon[other_word]: # Take word length into account here (instead of below, where act of lexicon words is determined)
                bigrams_common = []
                bigrams_append = bigrams_common.append
                bigram_overlap_counter = 0
                for bigram in range(len(lexicon_word_bigrams[lexicon[word]])):
                    if lexicon_word_bigrams[lexicon[word]][bigram] in lexicon_word_bigrams[lexicon[other_word]]:
                        bigrams_append(lexicon_word_bigrams[lexicon[word]][bigram])
                        lexicon_word_bigrams_set[lexicon[word]] = set(
                            lexicon_word_bigrams[lexicon[word]])
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

                # if word or other word is larger than the initial lexicon
                # (without PSC), overlap counter = 0, because words that are not
                # known should not inhibit
                if word >= n_known_words or other_word >= n_known_words:
                    total_overlap_counter = 0
                min_overlap = pm.min_overlap  # MM: currently 2

                ### NV: 2. take care of affix system, if relevant
                
                if pm.affix_system:

                    # NV: affixes dont exert inhibition on normal words, and between each other
                    if (lexicon[word] in affixes) or (lexicon[other_word] in affixes):
                        affix_only = True  # marks whether one of 2 words is an affix, useful for later
                        total_overlap_counter = 0
                    else:
                        affix_only = False

                    # if word or other-word is not only an affix itself, and the 2 words arent the same
                    if not(affix_only) and lexicon[word] != lexicon[other_word]:

                        # get stem from full word (for ex., weaken should output weak as inferred stem)
                        inferred_stem_otherword, matching_otherword = extract_stem(
                            lexicon[other_word], prefixes, suffixes, affixes)
                        inferred_stem_word, matching_word = extract_stem(
                            lexicon[word], prefixes, suffixes, affixes)

                        # if word is affixed (matching contains affixes)
                        # NV: determine if word-stem distance is within threshold, given max allowed edit distance, edit distance algorithm,
                        # and cutoff (under cutoff (short words), stem and word must be exactly the same.)
                        # here, we determined best values to be max_edit_dist = 1, cutoff=3, with algo = lcs.
                        # cutoff 4 yields slightly better precision, for slightly worse recall.
                        if (any(matching_otherword) and len(inferred_stem_otherword) > 1) and \
                            word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                            lexicon[word].strip('_'), inferred_stem_otherword):
                            complex_stem_pairs.append(
                                (lexicon[other_word], lexicon[word]))  # order:complex-stem (weaken, weak)

                        elif (any(matching_word) and len(inferred_stem_word) > 1) and \
                            word_stem_match(pm.simil_algo, pm.max_edit_dist, pm.short_word_cutoff,
                                            lexicon[other_word].strip('_'), inferred_stem_word):
                            complex_stem_pairs.append(
                                (lexicon[word], lexicon[other_word]))  # order:complex-stem (weaken, weak)

                ### NV: 3. Set inhibition values
                
                if complete_selective_word_inhibition:  # NV: what does this do?
                    # NV: added, because i suppose a word does not inhibit itself.
                    if total_overlap_counter > min_overlap and word != other_word:
                        # NV: remove min overlap from total?
                        word_overlap_matrix[word,
                                            other_word] = total_overlap_counter - min_overlap
                        word_overlap_matrix[other_word,
                                            word] = total_overlap_counter - min_overlap
                    else:
                        word_overlap_matrix[word, other_word] = 0
                        word_overlap_matrix[other_word, word] = 0
                        
                else:  # is_similar_word_length
                    if total_overlap_counter > min_overlap:
                        word_inhibition_matrix[word, other_word] = True
                        word_inhibition_matrix[other_word, word] = True
                        overlap_list[word, other_word] = total_overlap_counter - min_overlap
                        overlap_list[other_word, word] = total_overlap_counter - min_overlap
                        sys.exit(
                            'Make sure to use slow version, fast/vectorized version not compatible')

                # also build matrix of total ngrams, to calculate overlap percentage
                #bigrams_sum = N_ngrams_lexicon[word]+N_ngrams_lexicon[other_word]
                #overlap_percentage_matrix[word, other_word] = total_overlap_counter/bigrams_sum
                #overlap_percentage_matrix[other_word, word] = total_overlap_counter/bigrams_sum

                # NV: #???: overlap_percentage_matrix not used later in script. commented out

        # NV: for relevant affixes, re-set inhib values to 0
        # mirrored: WEAKEN does not inhibit WEAK, and WEAK does not inhibit WEAKEN
        for word1, word2 in complex_stem_pairs:
            word_overlap_matrix[lexicon.index(word1), lexicon.index(word2)] = 0
            word_overlap_matrix[lexicon.index(word2), lexicon.index(word1)] = 0

        # Save overlap matrix, with individual words selected (why is this needed?)
        output_inhibition_matrix = 'Data/Inhibition_matrix_'+pm.short[pm.language]+'.dat'
        with open(output_inhibition_matrix, "wb") as f:
            pickle.dump(np.sum(word_overlap_matrix, axis=0)[individual_to_lexicon_indices], f)

        # NV: for performance analysis with different values of edit dist and cutoff.
        if pm.affix_system:
            with open(f'Data/word_stem_matching_results/complex_stem_pairs_{pm.simil_algo}_dist{pm.max_edit_dist}_cutoff{pm.short_word_cutoff}.dat', "wb") as f:
                pickle.dump(complex_stem_pairs, f)
        # NV: pickle whole matrix for next time
        with open('Data/Inhibition_matrix_previous.dat', "wb") as f:
            pickle.dump(word_overlap_matrix, f)
        # NV: save parameters of this matrix
        with open('Data/Inhib_matrix_params_latest_run.dat', "wb") as f:
            pickle.dump(str(lexicon_word_bigrams)+str(LEXICON_SIZE)+str(pm.min_overlap) +
                        str(complete_selective_word_inhibition)+str(n_known_words)+str(pm.affix_system) +
                        str(pm.simil_algo)+str(pm.max_edit_dist) + str(pm.short_word_cutoff)+str(size_of_file), f)

    print("Inhibition grid ready.")
    print("")
    print("BEGIN EXPERIMENT")
    print("")
    logger.info("Inhibition grid ready. BEGIN EXPERIMENT")

    if pm.visualise:
        Visualise_reading
        #???

    # BEGIN EXPERIMENT
    # loop over trials
    stim = pm.stim
    unrecognized_words = []  # NV: init empty list outside the for loop. Before, would only remember the last word

    for trial in range(0, len(stim['all'])):

        print("trial: " + str(trial+1))
        logger.info("trial: " + str(trial+1))

        all_data.append({})

        stimulus = stim['all'][trial]

        stimulus_padded = " " + stimulus + " "

        if pm.is_priming_task:

            prime = stim['prime'][trial]
            prime_padded = " " + prime + " "

        # NV: eye position seems to be simply set in the beginning, and not manipulated (saccade blindness, etc)
        EyePosition = len(stimulus)//2

        # NV: in the case of embedded words, the eye position will be fixed to the center of the prime in the prime cycles, later.
        AttentionPosition = EyePosition

        all_data[trial] = {'stimulus': [],
                           'prime': [],  # NV: added prime
                           'target': [],
                           'condition': [],
                           'cycle': [],
                           'lexicon activity per cycle': [],
                           'stimulus activity per cycle': [],
                           'target activity per cycle': [],
                           'bigram activity per cycle': [],
                           'ngrams': [],
                           # 'recognized words indices': [],
                           # 'attentional width': attendWidth,
                           'exact recognized words positions': [],
                           'exact recognized words': [],
                           'eye position': EyePosition,
                           'attention position': AttentionPosition,
                           'word threshold': 0,
                           'word frequency': 0,
                           'word predictability': 0,
                           'reaction time': [],
                           'correct': [],
                           'POS': [],
                           'position': [],
                           'inhibition_value': pm.word_inhibition,  # NV: info for plots in notebook
                           'wordlen_threshold': pm.word_length_similarity_constant,
                           'target_inhib': [],
                           'error_rate': 0}  # NV: info for plots in notebook

        shift = False

        # # Lexicon word measures
        lexicon_word_inhibition_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_total_input_np = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_change = np.zeros((LEXICON_SIZE), dtype=float)
        lexicon_word_activity_np = np.zeros((LEXICON_SIZE), dtype=float)
        crt_word_activity_np = 0
        
        #init activity matrix with min activity
        lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity

        if task in ("Sentence", 'Classification', 'Transposed'):
            target = stimulus.split(" ")[stim['target'][trial]-1]  
            all_data[trial]['item_nr'] = stim['item_nr'][trial]
            all_data[trial]['position'] = stim['target'][trial]
            all_data[trial]['POS'] = (POSdict[target] if pm.use_grammar_prob else None)

        elif task == "Flanker":
            target = (stimulus.split()[1] if len(stimulus.split()) > 1 else stimulus.split()[0])

        elif task == "EmbeddedWords":
            target = stim['target'][trial]
            all_data[trial]['prime'] = prime
            all_data[trial]['item_nr'] = stim['item_nr'][trial]

        # KM: Getting the columns for EmbeddedWords_German
        elif task == "EmbeddedWords_German":
            target = stim['target'][trial]
            all_data[trial]['prime'] = prime
            all_data[trial]['item_nr'] = stim['item_nr'][trial]

        # KM: Getting the columns for EmbeddedWords_French
        elif task == "EmbeddedWords_French":
            target = stim['target'][trial]
            all_data[trial]['prime'] = prime
            all_data[trial]['item_nr'] = stim['item_nr'][trial]

        # store trial info in all_data
        all_data[trial]['stimulus'] = stimulus
        all_data[trial]['target'] = target
        all_data[trial]['condition'] = stim['condition'][trial]

        # enter the cycle-loop that builds word activity with every cycle
        # (re)set variables before loop body
        cycle_for_RT = 0  # MM: used to compute RT
        cur_cycle = 0  # MM: current cycle (cycle counter)
        recognized = False
        falseguess = False
        grammatical = False
        identif = False
        # Variables that count the level of activation of nouns or verbs to be used in the Classification task
        noun_count = 0
        ver_count = 0
        highest = None  # NV: reset highest activation index
        stimulus_list = []
        POSrecognition = {}

        # NV: could be changed to a sequence of for loops : If curcycle in stimcycles, etc
        while cur_cycle < pm.totalcycles:
            
            logger.info('\n')

            # NV: during blank stimulus presentation at the beginning or at the end
            if cur_cycle < pm.blankscreen_cycles_begin or cur_cycle > pm.totalcycles-pm.blankscreen_cycles_end:

                if pm.blankscreen_type == 'blank':  # NV decide what type of blank screen to show
                    # NV: overwrite stimulus with empty string. Note: stimulus is not padded, but next function expects padded input, hence the name. (for the empty string it does not matter)
                    stimulus = ""
                    stimulus_padded = "  "
                    logger.debug("Stimulus: blank screen")

                elif pm.blankscreen_type == 'hashgrid':
                    stimulus = "#####"  # NV: overwrite stimulus with hash grid
                    stimulus_padded = " ##### "
                    logger.debug("Stimulus: hashgrid screen")

                elif pm.blankscreen_type == 'fixation cross':
                    stimulus = "+"
                    stimulus_padded = " + "
                    logger.debug("Stimulus: fixation cross")

            # NV: If we are in priming cycle, set stimulus to the prime
            elif pm.is_priming_task and cur_cycle < (pm.blankscreen_cycles_begin+pm.ncyclesprime):
                stimulus = prime  # NV: overwrite stimulus with prime
                stimulus_padded = prime_padded
                logger.debug("Stimulus: "+stimulus)  # NV: show what is the actual stimulus

            else:
                # NV: reassign to change it back to original stimulus after prime or blankscreen.
                stimulus = stim['all'][trial]
                stimulus_padded = " "+stimulus+" "
                logger.debug("Stimulus: " + stimulus)

            # MM: check len stim, then determine order in which words are matched to slots in stim
            # Words checked in order of attentwght of word. To ease computation, we assume eye& attend in center.
            # Then attentweight highest on middle, fixated word, then on word just right of fixation
            # NV: these lists should reset when stimulus changes or when its the first stimulus
            # therefore they are computed within the cycle loop   
            # NV: this structure refers to the recently added slot-matching mechanism, and does not figure in the original OB1 paper. refer to Martijn or Nathan for more info
            n_words_in_stim = len(stimulus.split())
            if (n_words_in_stim <2):
                # if stim 1 wrd, it is checked first (note, indexing starts at 0!)
                order_match_check = [0]
            elif (n_words_in_stim == 2):
                # if stim 2 wrds, fst check right wrd, then left
                order_match_check = [1, 0]
            elif (n_words_in_stim == 3):
                # if stim 3 wrds, fst check middle wrd, then right, then left
                order_match_check = [1, 2, 0]
            elif (n_words_in_stim == 4):
                order_match_check = [2, 1, 3, 0]
            elif (n_words_in_stim == 5):
                order_match_check = [2, 3, 1, 4, 0]
            elif (n_words_in_stim == 6):
                order_match_check = [3, 2, 4, 1, 5, 0]
            elif (n_words_in_stim > 6):  # if more than 6 wrds, only consider fst 7
                order_match_check = [3, 4, 2, 5, 1, 6, 0]

            # NV: keep track of previous stimuli
            stimulus_list.append(stimulus)
            
            #TODO: Martijn: nadenken over wat er gebeurt als stimulus verandert
            if (len(stimulus_list) <= 1 ) or (stimulus_list[-2] != stimulus_list[-1]):
                
                # Now create list that will hold the recognized words
                stim_matched_slots = [""] * n_words_in_stim
                
                # Dictionary with a slot for each word in the stimulus, where we assign POS of the recognized
                # word in that position, or keep it blank if no word is recognized
                for slot_to_check in range(0, len(stimulus.split())):
                    POSrecognition[slot_to_check] = ''

            # NV: in elke cycle herbouw je de hele Ngram lijst, terwijl dat maar 1 keer hoeft (of 2 keer in priming task)
            (allNgrams, bigramsToLocations) = stringToBigramsAndLocations(
                stimulus_padded, is_prefix=False, is_suffix=False)
            # NV: set eye position in the middle of whatever is the stimulus
            EyePosition = len(stimulus)//2
            AttentionPosition = EyePosition
            allMonograms = []
            allBigrams = []

            for ngram in allNgrams:
                if len(ngram) == 2:
                    allBigrams.append(ngram)
                else:
                    allMonograms.append(ngram)
            allBigrams_set = set(allBigrams)
            logger.debug(allBigrams)

            # Reset
            unitActivations = {}
            word_input_np.fill(0.0)
            lexicon_word_inhibition_np.fill(0.0)
            lexicon_word_inhibition_np2.fill(0.0)
            lexicon_activewords_np.fill(False)

            # Calculate ngram activity
            # MM: could also be done above at start fix, and then again after attention shift. is constant in btw shifts
            for ngram in allNgrams:
                if len(ngram) == 2:
                    unitActivations[ngram] = calcBigramExtInput(ngram,
                                                                bigramsToLocations,
                                                                EyePosition,
                                                                AttentionPosition,
                                                                pm.attendWidth,
                                                                shift,
                                                                cur_cycle)
                elif len(ngram) == 1:
                    unitActivations[ngram] = calcMonogramExtInput(ngram,
                                                                  bigramsToLocations,
                                                                  EyePosition,
                                                                  AttentionPosition,
                                                                  pm.attendWidth,
                                                                  shift,
                                                                  cur_cycle)
                else:
                    raise RuntimeError("?")

            all_data[trial]['bigram activity per cycle'].append(sum(unitActivations.values()))
            logger.debug(f'bigram activity per cycle: {sum(unitActivations.values())}')
            all_data[trial]['ngrams'].append(len(allNgrams))

            # activation of word nodes

            # taking nr of ngrams, word-to-word inhibition etc. into account
            wordBigramsInhibitionInput = 0
            for ngram in allNgrams:
                wordBigramsInhibitionInput += pm.bigram_to_word_inhibition * \
                    unitActivations[ngram]

            # This is where input is computed (excit is specific to word, inhib same for all)
            for lexicon_ix, lexicon_word in enumerate(lexicon):  # NS: why is this?
                wordExcitationInput = 0

                # (Fast) Bigram & Monogram activations
                bigram_intersect_list = allBigrams_set.intersection(
                    lexicon_word_bigrams[lexicon_word])
                for bigram in bigram_intersect_list:
                    wordExcitationInput += pm.bigram_to_word_excitation * \
                        unitActivations[bigram]
                for monogram in allMonograms:
                    if monogram in lexicon_word:
                        wordExcitationInput += pm.bigram_to_word_excitation * \
                            unitActivations[monogram]

                word_input_np[lexicon_ix] = wordExcitationInput + wordBigramsInhibitionInput

            # divide input by nr ngrams (normalize)
            word_input_np = word_input_np / np.array(N_ngrams_lexicon)

            # NV: by now, all bigram excitation/inhibition effects are calculated. Hereunder, inter-word inhibition is adressed.

            # Active words selection vector (makes computations efficient)
            lexicon_activewords_np[(lexicon_word_activity_np > 0.0) | (word_input_np > 0.0)] = True

            # Calculate total inhibition for each word
            # Matrix * Vector (4x faster than vector)
            overlap_select = word_overlap_matrix[:, (lexicon_activewords_np == True)]

            lexicon_select = (lexicon_word_activity_np+word_input_np)[(
                lexicon_activewords_np == True)] * lexicon_normalized_word_inhibition  # NV: the more active a certain word is, the more inhibition it will execute on its peers -> activity is multiplied by inhibition constant.
            # NV: then, this inhibition value is weighed by how much overlap there is between that word and every other. BUT! longer words will have more overlap, and will be more inhibited. Should that be corrected?

            # NV: Overlap and activity are now squared before applying dot product.
            # This concentrates inhibition on the words that have most overlap and are most active. 
            # As a result, irrelavant words play a smaller role in the inhibition of a word.
            lexicon_word_inhibition_np = np.dot((overlap_select**2), -(lexicon_select**2)) / np.array(N_ngrams_lexicon)
            
            #Alternative: select 10 smallest values to retain for inhibition. will concentrate inhib. on the most relevant words
            #lexicon_word_inhibition_np = np.array([np.partition((overlap_select[index_num1,:] * lexicon_select), 10, axis=None)[:10].sum()
            #                             for index_num1 in range(len(overlap_select))]) / np.array(N_ngrams_lexicon) 
            
            # Combine word inhibition and input, and update word activity
            lexicon_total_input_np = np.add(
                lexicon_word_inhibition_np, word_input_np)

            # now comes the formula for computing word activity.
            # pm.decay has a neg value, that's why it's here added, not subtracted
            lexicon_word_activity_change = ((pm.max_activity - lexicon_word_activity_np) * lexicon_total_input_np) + \
                                        ((lexicon_word_activity_np - pm.min_activity) * pm.decay)
            lexicon_word_activity_np = np.add(lexicon_word_activity_np, lexicon_word_activity_change)

            # Correct activity beyond minimum and maximum activity to min and max
            lexicon_word_activity_np[lexicon_word_activity_np < pm.min_activity] = pm.min_activity
            lexicon_word_activity_np[lexicon_word_activity_np > pm.max_activity] = pm.max_activity
            
            # Save current word activities (per cycle)
            target_lexicon_index = individual_to_lexicon_indices[[
                idx for idx, element in enumerate(lexicon) if element == '_'+target+'_']]
            logger.debug("target index:" + str(target_lexicon_index))
            
            #plot_inhibition_matrix 
            if pm.plotting and trial == 100:
                
                #TODO: cleanup and move to plotting function
                index_num1=lexicon.index('_drag_')
                inhib_spectrum1 = np.sort((overlap_select[index_num1,:] * lexicon_select))[:10] \
                    / N_ngrams_lexicon[index_num1]
                inhib_spectrum1_indices = np.argsort((overlap_select[index_num1,:] * lexicon_select))[:10]
                
                index_num2=lexicon.index('_dragon_')
                inhib_spectrum2 = np.sort((overlap_select[index_num2,:] * lexicon_select))[:10] \
                    / N_ngrams_lexicon[index_num2]
                inhib_spectrum2_indices = np.argsort((overlap_select[index_num2,:] * lexicon_select))[:10]
                
                plot_inhib_spectrum(lexicon, lexicon_activewords_np, inhib_spectrum1, inhib_spectrum2,
                                    index_num1, index_num2, inhib_spectrum1_indices, inhib_spectrum2_indices, cur_cycle)
                
            # only plot when not blanscreen/mask. also plot only one trial, to not clutter the plot space
            if pm.plotting and cur_cycle >= pm.blankscreen_cycles_begin  and cur_cycle <= pm.totalcycles-pm.blankscreen_cycles_end and trial in (5, 88, 49):

                plot_runtime(stimulus, N_ngrams_lexicon, lexicon_activewords_np,
                             lexicon_word_inhibition_np, word_input_np, lexicon_word_activity_np,
                             lexicon_thresholds_np, lexicon, [lexicon.index(f'_{prime}_'), lexicon.index(f'_{target}_')])

                logger.info('done plotting')
        
                        
            all_data[trial]['target_inhib'].append(lexicon_word_inhibition_np[target_lexicon_index])
            
            
            #crt_word_total_input_np = lexicon_total_input_np[target_lexicon_index]
            crt_word_activity_np = lexicon_word_activity_np[target_lexicon_index]
            all_data[trial]['target activity per cycle'].append(crt_word_activity_np)
        
            logger.debug("target activity:" + str(crt_word_activity_np))
            logger.debug("target threshold:" + str(lexicon_thresholds_np[target_lexicon_index]))
            
            
            # Alternative measure for N400: just act of wrds in stim
            # "stimulus activity" is now computed by adding the activations for each word (target and flankers) in stimulus
            # NV: as words are saved as _word_ in lexicon, look up stimulus with added _'s
            stim_activity = sum([lexicon_word_activity_np[lexicon_index_dict['_'+word+'_']]
                                 for word in stimulus.split() if '_'+word+'_' in lexicon])
            all_data[trial]['stimulus activity per cycle'].append(
                stim_activity)
            logger.debug("stimulus activity:" + str(stim_activity))

            # MM: change tot act to act in all lexicon
            total_activity = sum(lexicon_word_activity_np)
            all_data[trial]['lexicon activity per cycle'].append(total_activity)
            logger.debug("total activity: "+str(total_activity))

            # Enter any recognized word to the 'recognized words indices' list
            # creates array (MM: msk?) that is 1 if act(word)>thres, 0 otherwise
            above_thresh_lexicon_np = np.where(
                lexicon_word_activity_np > lexicon_thresholds_np, 1, 0)

            all_data[trial]['cycle'].append(cur_cycle)

            all_data[trial]['exact recognized words positions'].append(
                np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0][:])
            all_data[trial]['exact recognized words'].append(
                [lexicon[i] for i in np.where(lexicon_word_activity_np > lexicon_thresholds_np)[0]])

            logger.debug("nr. above thresh. in lexicon: " + str(np.sum(above_thresh_lexicon_np)))

            # NV: print words that are above threshold
            words_above_threshold = [x for i, x in enumerate(
                lexicon) if above_thresh_lexicon_np[i] == 1]
            logger.debug("recognized words " + str(words_above_threshold))

            # keep list of words recognized in this cycle
            new_recognized_words = np.zeros(LEXICON_SIZE)
                            
            # MM: We now check whether words in stim are recognized, by checking matching active wrds to slots
            # We check the n slot in order order_match_check
            for slot_to_check in range(0, n_words_in_stim):
                # slot_num is the slot in the stim (spot of still-unrecogn word) that we're checking
                slot_num = order_match_check[slot_to_check]
                # if the slot has not yet been filled..
                if len(stim_matched_slots[slot_num]) == 0:
                    
                    # Check words that have the same length as word in the slot we're now looking for
                    word_searched = stimulus.split()[slot_num]
                    
                    # MM: recognWrdsFittingLen_np: array with 1=wrd act above threshold, & approx same len
                    # as to-be-recogn wrd (with 15% margin), 0=otherwise
                    # NV: exclude affixes to be recognized as words                    
                    recognWrdsFittingLen_np = above_thresh_lexicon_np * \
                        np.array([0 if x in affixes else int(is_similar_word_length(len(x.replace('_', '')),
                                  [len(word_searched)])) for x in lexicon])

                    # Fast check whether there is at least one 1 in wrdsFittingLen_np (thus above thresh.)
                    if sum(recognWrdsFittingLen_np): 

                        # Find the word with the highest activation in all words that have a similar length
                        # and recognise that word's POS
                        highest = np.argmax(recognWrdsFittingLen_np * lexicon_word_activity_np)
                        highest_word = lexicon[highest]
                        
                        logger.info(f"""word {highest_word.replace('_', '')} matched in slot {slot_num}""")

                        # The winner is matched to the slot, and its activity is reset to minimum to not have it matched to other words
                        stim_matched_slots[slot_num] = highest_word
                        new_recognized_words[highest] = 1
                        above_thresh_lexicon_np[highest] = 0
                        lexicon_word_activity_np[highest] = pm.min_activity
                        
                        #check if target is on screen first
                        if target in stimulus.split():
                            
                            #if we are considering the target slot
                            if stimulus.split().index(target) == slot_num:
                                
                                #if we recognized the target in the target slot: stop trial
                                if target == highest_word.replace('_', ''):
                                    recognized = True
                                    logger.info('matched word is target word')
                                
                                #recognized another word in target slot: stop trial
                                else:
                                    logger.info('matched word is not target word, but should have been')
                                    falseguess = True
                            
                            #recognized a word in a non-target-slot: continue trial
                            else:
                                logger.info('slot is not target slot')
                        
                        #recognized a word while target was not in stimulus: continue trial
                        else:
                            logger.info('prime recognized')


                        if pm.use_grammar_prob:

                            POSrecognition[slot_num] = POSdict[highest_word.replace('_', '')]
                            logger.info('Word searched:'+ word_searched+ ' Highest:'+
                                  highest_word+ 'Number highest:'+ str(highest))

                            # For the classification task, if one of the flankers is recognized as a noun or verb,
                            # then we add the word's activity to the count
                            # NV: add _ to word
                            if task == 'Classification':
                                if POSrecognition[0] == 'NOU' or POSrecognition[2] == 'NOU':
                                    noun_count += lexicon_word_activity_np[lexicon_index_dict[
                                        f'_{stimulus.split()[ slot_num]}_']]
                                elif POSrecognition[0] == 'VER' or POSrecognition[2] == 'VER':
                                    ver_count += lexicon_word_activity_np[lexicon_index_dict[
                                        f'_{stimulus.split()[ slot_num]}_']]

                            # If we matched the right POS (indep of whether the target was recognized), add grammar probs (actually lazy implementation, because grammar should also play role
                            # when wrong POS is recogn - but assume for now that wrong word has rnd category, so grammar effects will cancel out)
                            # Add the effect of grammar to the activation of the words
                            if POSrecognition[slot_num] == POSdict[stimulus.split()[slot_num]]:
                                # Add more activity to words before and after, according to grammar probabilties
                                if slot_num > 0:
                                    # Add word_pred times grammar_wgt to activ of word preceding matched word
                                    lexicon_word_activity_np[lexicon_index_dict[f'_{stimulus.split()[slot_num - 1]}_']]\
                                        += word_pred_values[0][trial][slot_num - 1] * grammar_weight
                                if slot_num < len(stimulus.split())-1:
                                    # Add word_pred times grammar_wgt to activ of word following matched word
                                    lexicon_word_activity_np[lexicon_index_dict[f'_{stimulus.split()[slot_num + 1]}_']] \
                                        += word_pred_values[1][trial][slot_num + 1] * grammar_weight

            if task == 'Transposed':
                # When the 3 centre words are recognized, we can decide if the sentence is grammatical or not
                if POSrecognition[1] != '' and POSrecognition[2] != '' and POSrecognition[3] != '':

                    if stim['condition'][trial] == 'normal':
                        grammatical = True

                    elif stim['condition'][trial] == 'within23' or stim['condition'][trial] == 'across23':
                        # Since since slot_num starts at 0, the word in position 2 is in slot_num 1
                        # If the POSpair of the transposed words has a grammatical probability higher than
                        # the threshold, then we consider the sentence grammatical
                        POSpair = POSrecognition[1] + POSrecognition[2]
                        # Threshold for considering a sentence grammatical
                        if (Grammardict[POSpair][0] + Grammardict[POSpair][1])/2 > 0.19:
                            grammatical = False
                    elif stim['condition'][trial] == 'within34' or stim['condition'][trial] == 'across34':
                        POSpair = POSrecognition[2] + POSrecognition[3]
                        if (Grammardict[POSpair][0] + Grammardict[POSpair][1])/2 > 0.19:
                            grammatical = False

            if task == 'Classification':
                if POSrecognition[0] != '' and POSrecognition[2] != '':

                    # If the activity of the nouns or verbs in the sentence is higher than the
                    # threshold, then we are biased to recognize the target as having the same POS
                    if noun_count > 1.5:
                        POSrecognition[1] = 'NOU'
                    elif ver_count > 1.5:
                        POSrecognition[1] = 'VER'
                    # If the recognized POS is the same as the target's POS, then it was classified correctly
                    if POSrecognition[1] == POSdict[target]:
                        identif = True

            # MM: bit odd but smart way to compute moment of recogn: each time step that tehre is no recogn, you set cycle_recogn to the current cycle
            # This stops when target is recognized
            if recognized == False:
                cycle_for_RT = cur_cycle
            
            #NV: if the design of the task considers the first recognized word in the target slot to be the final response, stop the trial when this happens
            if pm.trial_ends_on_key_press and (recognized == True or falseguess == True):

                # NV: this is just a little check if the matrix of recognized words is balanced. If eveything is <0, the highest word still gest picked out, but the analysis via Noor's notebook will not be sucessful
                # [element for element in lexicon_total_input_np if element>0]
                check = any(lexicon_total_input_np > 0)
                if not check:
                    print(
                        'WARNING: all word activations are negative. make sure inhibition/excitation balance in parameters is ok. You can set pm.plotting to True to see the inhibition values during the task')
                    logger.warning(
                        'all word activations are negative. make sure inhibition/excitation balance in parameters is ok. You can set pm.plotting to True to see the inhibition values during the task')

                break

            cur_cycle += 1

        # NV: determine if word is recognized or not. Special mechanism for Transposed or Clasification, default method otherwise.
        if task == 'Transposed':
            if (grammatical == True and stim['condition'][trial] != 'normal') or (grammatical == False and stim['condition'][trial] == 'normal'):
                all_data[trial]['correct'].append(0)
            else:
                all_data[trial]['correct'].append(1)

        elif task == 'Classification':
            if identif == False:
                all_data[trial]['correct'].append(0)
            else:
                all_data[trial]['correct'].append(1)
        else:
            if recognized == False:
                unrecognized_words.append(target)
                all_data[trial]['correct'].append(0)
            else:
                all_data[trial]['correct'].append(1)

        # MM: CHECK WHAT AVERAGE NON-DECISION TIME IS? OR RESPONSE EXECUTION TIME?
        reaction_time = ((cycle_for_RT+1-pm.blankscreen_cycles_begin) * pm.CYCLE_SIZE)+300
        print("reaction time: " + str(reaction_time) + " ms")
        logger.info("reaction time: " + str(reaction_time) + " ms")
        all_data[trial]['reaction time'].append(reaction_time)
        all_data[trial]['word threshold'] = word_thresh_dict.get(target, "")
        all_data[trial]['word frequency'] = word_freq_dict.get(target, "")
        # NV: added error info for plotting later. divide number of wrong words by total trials -> error rate. Added max(1,trial) to avoid divide by 0
        all_data[trial]['error_rate'] = str(len(unrecognized_words)/max(1, trial))

        print("end of trial")
        print("----------------")
        print("\n")
        logger.info("end of trial \n")

    # END OF EXPERIMENT. Return all data and a list of unrecognized words
    return lexicon, all_data, unrecognized_words


if __name__ == '__main__':
    pm = return_params()
    (lexicon, all_data, unrecognized_words) = simulate_experiments('EmbeddedWords', pm)

	