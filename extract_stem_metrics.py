# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 10:54:47 2022

@author: nvs690
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

algos = {}
# get stem lists for different algorithms
directory = 'Data/word_stem_matching_results'
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):
        varname = filename.strip(".dat")
        with open(f, 'rb') as f2:
            temp = pickle.load(f2)
            algos[varname] = [(word1.strip('_'), word2.strip('_')) for (word1, word2) in temp]

EW = pd.read_csv('Stimuli/EmbeddedWords_stimuli_all_csv.csv')
options = ['truly suffixed/related prime', 'pseudo suffixed/related prime']
# get all words pairs that have an affix
reference_pairs = EW.loc[:, ['prime', 'target']][EW["condition"].isin(options)] 

#df to tuple list (for set operations)
rp_tuples = []
for item in reference_pairs.values:
    rp_tuples.append((item[0].strip('_'), item[1].strip('_')))

results = {}
for algo_name, algo_res in algos.items():
    dic1 = {}
    # intersection is the pairs that is both in the reference, and has been found by the algo
    inter1 = set(rp_tuples).intersection(algo_res)
    inter2 = set([(word2, word1) for (word1, word2) in rp_tuples]
                 ).intersection(algo_res)  # also look in reverse tuples
    inter=max((inter1, inter2), key=len) # the longest is the intended matches. Other is artifact
    
    truepos = len(inter)
    falsepos = len(algo_res)-len(inter)  # all positives minus true positives
    falseneg = len(rp_tuples)-len(inter)  # tuples not found

    dic1['precision'] = truepos/(truepos+falsepos)
    dic1['recall'] = truepos/(truepos+falseneg)
    dic1['falsepositives'] = set(algo_res).difference(inter)
    # TODO: hardcoded, because we know rp_tuples is always the inverse order than algo_res. But be careful here.
    dic1['falsenegatives'] = set(rp_tuples).difference(inter)
    results[algo_name] = dic1


plt.rcParams["figure.figsize"] = (6,6)
for algo_name, algo_res in results.items():
    plt.plot(algo_res["precision"], algo_res["recall"], ['o' if 'lev' in algo_name else 'x' if 'lcs' in algo_name else 's'][0], label=algo_name[19:])
plt.xlabel('precision')
plt.ylabel('recall')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.legend()
plt.show()