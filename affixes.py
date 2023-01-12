#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:02:01 2021

@author: nathanvaartjes

The purpose of this script is to build a dictionnary of affixes and their log frequency per million (zipf).
This dictionary is then pickled to later be fetched when needed.
Adapted from Jarmulowicz et al.(2002)(DOI: 10.1006/brln.2001.2517 ) NOTE: -ly not counted as suffix in this paper.
Necessary for the simulation for the Embedded Words task (Beyersmann 2020)

"""
import numpy as np
import pandas as pd
import pickle
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import string

def create_affix_files():
    
    
    prefixes_EN = pd.ExcelFile('Texts/MorphoLEX_en.xlsx').parse('All prefixes')
    suffixes_EN = pd.ExcelFile('Texts/MorphoLEX_en.xlsx').parse('All suffixes')
    prefixes_FR = pd.ExcelFile('Texts/Morpholex_FR.xlsx').parse('prefixes')
    suffixes_FR = pd.ExcelFile('Texts/Morpholex_FR.xlsx').parse('suffixes')
    print(prefixes_EN)
    
    # column has no name in Morpholex_FR: add it
    prefixes_FR.rename(columns={"Unnamed: 0": "morpheme"}, inplace=True)
    suffixes_FR.rename(columns={"Unnamed: 0": "morpheme"}, inplace=True)
    
    # cleanup: remove punctuation: >en> becomes en
    prefixes_EN["morpheme"] = prefixes_EN["morpheme"].str.replace(f'[{string.punctuation}]', '')
    suffixes_EN["morpheme"] = suffixes_EN["morpheme"].str.replace(f'[{string.punctuation}]', '')
    prefixes_FR["morpheme"] = prefixes_FR["morpheme"].str.replace(f'[{string.punctuation}]', '')
    
    # suffixes FR: also have to remove [VB] such as in >er>[VB]
    suffixes_FR["morpheme"] = suffixes_FR["morpheme"].str.replace("[VB]", "")
    suffixes_FR["morpheme"] = suffixes_FR["morpheme"].str.replace(f'[{string.punctuation}]', '')
    
    
    to_remove = []
    for ix, row in suffixes_FR.iterrows():
        # ant/ent became antent after punct removal, so keep ant and add ent at the end of df
        if row["morpheme"] == 'antent':
            suffixes_FR.at[ix, "morpheme"] = 'ant'
            to_append = copy.deepcopy(row)
            to_append["morpheme"] = 'ent'
        # remove words that are now empty (were only [VB] or punctuation)
        if row["morpheme"] == '':
            to_remove.append(ix)
    
    suffixes_FR.drop(index=to_remove, inplace=True)
    suffixes_FR = suffixes_FR.append(to_append, ignore_index=True)
    
    
    # EN: convert from HAL to Zipf
    prefixes_EN["Zipf_freq"] = [0 if x == 0 else np.log10((x/400000000)*1E9)
                                for x in prefixes_EN["HAL_freq"]]
    suffixes_EN["Zipf_freq"] = [0 if x == 0 else np.log10((x/400000000)*1E9)
                                for x in suffixes_EN["HAL_freq"]]
    
    # FR: convert from freq per million to Zipf
    prefixes_FR["Zipf_freq"] = [0 if x == 0 else np.log10(x*1000) for x in prefixes_FR["summed_freq"]]
    suffixes_FR["Zipf_freq"] = [0 if x == 0 else np.log10(x*1000) for x in suffixes_FR["summed_freq"]]
    
    # make dict and pickle
    prefixes_EN_dict = dict(zip(prefixes_EN["morpheme"], prefixes_EN["Zipf_freq"]))
    with open('Data/prefix_frequency_en.dat', 'wb') as f:
        pickle.dump(prefixes_EN_dict, f)
    
    suffixes_EN_dict = dict(zip(suffixes_EN["morpheme"], suffixes_EN["Zipf_freq"]))
    with open('Data/suffix_frequency_en.dat', 'wb') as f:
        pickle.dump(suffixes_EN_dict, f)
    
    prefixes_FR_dict = dict(zip(prefixes_FR["morpheme"], prefixes_FR["Zipf_freq"]))
    with open('Data/prefix_frequency_fr.dat', 'wb') as f:
        pickle.dump(prefixes_FR_dict, f)
    
    suffixes_FR_dict = dict(zip(suffixes_FR["morpheme"], suffixes_FR["Zipf_freq"]))
    with open('Data/suffix_frequency_fr.dat', 'wb') as f:
        pickle.dump(suffixes_FR_dict, f)
    
    
    diag_plots = True
    
    if diag_plots == True:
    
        pdconcat = pd.concat([prefixes_EN["Zipf_freq"],
                              suffixes_EN["Zipf_freq"],
                              prefixes_FR["Zipf_freq"],
                              suffixes_FR["Zipf_freq"]],
                             keys=['pre_EN', 'suf_EN', 'pre_FR', 'suf_FR'], 
                             names=['Series name', 'Row ID']).to_frame()
        sns.histplot(pdconcat, x="Zipf_freq", hue='Series name', element="step")
        plt.show()
    
    
    # old data (Jarmulowicz, 2002), kept for reference for now
    
    # suffix_totalcount_en = {'tion': 122,
    #                         'ion': 122,
    #                         'al': 91,
    #                         'ial': 91,
    #                         'er': 85,
    #                         'y': 75,
    #                         'ment': 59,
    #                         'ous': 51,
    #                         'ious': 51,
    #                         'ant': 50,
    #                         'ent': 50,
    #                         'an': 48,
    #                         'ian': 48,
    #                         'ar': 43,
    #                         'or': 43,
    #                         'ance': 30,
    #                         'ence': 30,
    #                         'ity': 28,
    #                         'able': 24,
    #                         'ible': 24,
    #                         'ate': 22,
    #                         'ful': 22,
    #                         'ive': 17,
    #                         'ice': 16,
    #                         'ise': 16,
    #                         'ic': 13,
    #                         'en': 13,
    #                         'ship': 11,
    #                         'ure': 11,
    #                         'ness': 10,
    #                         'ern': 8,
    #                         'age': 8,
    #                         'ize': 8,
    #                         'less': 6,
    #                         'ism': 5,
    #                         'ary': 5,
    #                         'th': 4,
    #                         'ite': 3,
    #                         'ist': 3,
    #                         'cracy': 2,
    #                         'ide': 1,
    #                         'hood': 1,
    #                         'ify': 1}
    
    # suffix_zipf = {}
    
    # for i, (j, k) in enumerate(suffix_totalcount_en.items()):
    #     # log10 of frequency per billion (24680 words in text in total)
    #     suffix_zipf[j] = np.log10((k/24680)*1E9)
    
    # with open('Data/suffix_frequency_en.dat', 'wb') as f:
    #     pickle.dump(suffix_zipf, f)
    
    


def get_suffix_file(pm): #NV: added function to read affixes frequency data from pickle, written by affixes.py. Which affixes to fetch is independent of task but dependent on language.
    file="Data/suffix_frequency_"+pm.short[pm.language]+".dat"
    with open (file,"rb") as f:
       suffix_freq_dict = pickle.load(f)
       print(f'Loading this file for suffixes: {file}')
       return suffix_freq_dict
       #NV: predictability does not really make sense in the context of affixes, hence they are not made nor imported
       #But maybe something to look at in the future

def get_prefix_file(pm): #NV: added function to read affixes frequency data from pickle, written by affixes.py. Which affixes to fetch is independent of task but dependent on language.
    file="Data/prefix_frequency_"+pm.short[pm.language]+".dat"
    with open (file,"rb") as f:
       prefix_freq_dict = pickle.load(f)
       return prefix_freq_dict
       #NV: predictability does not really make sense in the context of affixes, hence they are not made nor imported
       #But maybe something to look at in the future
       
if __name__ == '__main__':
    #when running this file directly, create affix files. Other functions are meant to be imported in other modules.
    create_affix_files()