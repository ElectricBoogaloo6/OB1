from __future__ import division
from reading_common import get_stimulus_text_from_file, get_stimulus_text_from_file2
import pylab as p
import math
import codecs
import re
import pandas as pd
import read_saccade_data as exp
import pdb
import numpy as np
import random
import itertools
__author__ = 'Sam van Leipsig'




## Basic Parse
def create_df_psc(filepath_psc):
    parsed_psc = get_stimulus_text_from_file(filepath_psc)
    psc_list =  parsed_psc.split(" ")
    df_psc = pd.DataFrame(psc_list,columns=['words'])
    df_psc['word length'] = df_psc['words'].map(lambda x: len(x))
    df_freq_pred = pd.DataFrame(exp.get_freq_and_pred())
    df_freq_pred = df_freq_pred.iloc[0:len(df_psc),:]
    df_psc = pd.concat([df_psc,df_freq_pred],axis=1,join_axes=[df_psc.index])
    df_psc = df_psc.drop(['word'],1)
    df_psc.rename(columns={'f':'freq'}, inplace=True)
    df_psc['index value'] = df_psc.index
    df_psc['control words'] = np.zeros(len(df_psc['words']))
    df_psc['control indices'] = np.zeros(len(df_psc['words']))
    df_psc['sentence number'] = np.zeros(len(df_psc['words']))
    df_psc['condition type'] = np.zeros(len(df_psc['words']))
    df_psc = df_psc.drop(df_psc.index[-1])
    return df_psc

def parse_sentences(filepath_psc):
    sentence_lenghts_list = []
    for sentence in get_stimulus_text_from_file2(filepath_psc).replace(',','').split('.'):
        sentence_list = sentence.strip(" ")
        sentence_list = sentence_list.split(" ")
        if len(sentence_list) > 1: ## ugly hardcode to remove list with only space
            myarray = np.array(sentence_list,dtype='U')
            sentence_lenghts_list.append(len(myarray))
    return sentence_lenghts_list

def remove_capitals(df_psc):
    df_psc['words'] = df_psc['words'].apply(lambda x: x.lower())
    return df_psc

def add_sentence_numbers(df_psc, sentence_lenghts_list):
    sumprevious = 0
    sentence_number = 1
    for sentence_length in sentence_lenghts_list:
        for x in range(0,sentence_length,1):
            df_psc.loc[x+sumprevious,['sentence number']] = sentence_number
        sumprevious+=sentence_length
        sentence_number+=1
    return df_psc


def select_condition_words(df_psc,length_selection,max_freq_diff,max_pred):
    index_list = []
    for name,wl_group in df_psc.groupby('word length'):
        if name in length_selection:
            for i in xrange(0,len(wl_group.index)-2):
                if (wl_group.index[i]+1 == wl_group.index[i+1]) and abs(wl_group.loc[wl_group.index[i],:]['freq'] - wl_group.loc[wl_group.index[i+1],:]['freq']) < max_freq_diff:
                    if  wl_group.loc[wl_group.index[i+1],:]['pred'] < max_pred:
                        index_list.append(wl_group.index[i])
    df_psc_selection = df_psc.iloc[index_list]
    df_psc_selection = df_psc_selection[df_psc_selection['pred'] < max_pred]
    df_psc_selection =  df_psc_selection[~df_psc_selection['words'].duplicated()]
    return df_psc_selection


def comparestring(word1,word2):
    limit = 0
    for monogram in word1:
        if monogram in word2:
            limit+=1
    if limit == 0:
        return True
    else:
        return False

def comparestring2(word1,word2):
    limit = 0
    for monogram in word1:
        if monogram in word2:
            limit+=1
    return limit

def add_control_words(df_psc,df_psc_selection,length_selection,max_freq_diff):
    ## Select control words
    for word_length in length_selection:
        df_psc_selection_wl = df_psc_selection[df_psc_selection['word length']==word_length]
        df_psc_wl = df_psc[df_psc['word length']==word_length]
        for i in xrange(len(df_psc_selection_wl['freq'])):
            freqstart = df_psc_selection_wl.iloc[i]['freq'] - max_freq_diff
            freqend = df_psc_selection_wl.iloc[i]['freq'] + max_freq_diff
            #(df_psc_wl['freq'] - df_psc_selection_wl.iloc[i]['freq']).abs().argsort()[:10]
            freq_selection =  df_psc_wl[(df_psc_wl['word length']==word_length) & (df_psc_wl['freq']>freqstart) & (df_psc_wl['freq']<freqend)]
            string_selection =  freq_selection['words'].apply(lambda x: comparestring(x,df_psc_selection_wl.iloc[i]['words']))
            final_selection =  freq_selection[string_selection]
            if len(final_selection)>0:
                control_words = final_selection[~final_selection['words'].duplicated()]
                control_words = control_words.reset_index()
                baseline_word = df_psc.loc[df_psc_selection_wl.iloc[i]['index value']+1,['words']][0]
                control_words['overlap'] = np.zeros(len(control_words['words']))
                control_words['overlap'] = control_words['words'].apply(lambda controle_word: comparestring2(controle_word,baseline_word))
                min_overlap_base_control =  control_words['overlap'].min()
                control_words_best = control_words[control_words['overlap']==min_overlap_base_control]
                random_control_word = control_words_best.ix[random.sample(control_words_best.index, 1)]['words'].values[0]
                random_control_index = control_words_best.ix[random.sample(control_words_best.index, 1)]['index value'].values[0]

                ## Can select other control words
                df_psc_selection.loc[df_psc_selection_wl.iloc[i]['index value'],['control words']] = random_control_word
                df_psc_selection.loc[df_psc_selection_wl.iloc[i]['index value'],['control indices']] = random_control_index
                df_psc.loc[df_psc_selection_wl.iloc[i]['index value'],['control words']] = random_control_word
                df_psc.loc[df_psc_selection_wl.iloc[i]['index value'],['control indices']] = random_control_index
    return df_psc

def create_filler_sentences(df_psc,condition_sentence_numbers):
    all_sentences = np.arange(0,max(df_psc['sentence number']),1,dtype=int)
    mask = np.ones(len(all_sentences), np.bool)
    mask[condition_sentence_numbers] = 0
    filler_sentence_numbers = all_sentences[mask]
    filler_sentence_numbers_unique = set(filler_sentence_numbers)
    return random.sample(filler_sentence_numbers_unique, len(filler_sentence_numbers_unique))


def random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number):
    random_sentence = df_psc[df_psc['sentence number'] == filler_sentence_numbers[number]].reset_index(drop=True)
    return pd.concat([df_exp_psc,random_sentence],axis=0), (number+1)

def condition_sentence(i,condition_sentence_numbers,df_psc,df_exp_psc,condition_type):
    condition_sentence = df_psc[df_psc['sentence number'] == condition_sentence_numbers[i]].reset_index(drop=True)
    condition_sentence.ix[condition_sentence['control indices']!=0,['condition type']] = condition_type
    return pd.concat([df_exp_psc,condition_sentence],axis=0)

def make_df_boundarytask(condition_sentence_numbers,filler_sentence_numbers,df_psc,condition_repeats):
    df_exp_psc = pd.DataFrame()
    initial_number = random.randint(0,len(filler_sentence_numbers)-1)
    for i in range(1,5,1):
        df_exp_psc, initial_number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,initial_number)
    number = 0
    number_random_sentences = 7
    #orderlist = [ [1,2,3], [2,3,1], [2,1,3], [3,1,2], [3,2,1],[1,3,2] ]
    for test_order in itertools.permutations((1,2,3)):
        for i in range(0,len(condition_sentence_numbers),1):
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc = condition_sentence(i,condition_sentence_numbers,df_psc,df_exp_psc,test_order[0])
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc = condition_sentence(i,condition_sentence_numbers,df_psc,df_exp_psc,test_order[1])
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc, number = random_sentence(filler_sentence_numbers,df_psc,df_exp_psc,number)
            df_exp_psc = condition_sentence(i,condition_sentence_numbers,df_psc,df_exp_psc,test_order[2])
            if (number + number_random_sentences) > (len(filler_sentence_numbers)-1):
                number = 0
    df_exp_psc = df_exp_psc.reset_index()
    df_exp_psc.rename(columns={'index':'in sentence pos'}, inplace=True)
    return df_exp_psc




## TESTING
# df_psc_grouped = df_psc.groupby('word length')
#
# for name,group in df_psc_grouped:
#     start = group.index.searchsorted(df_psc_selection.iloc[0]['freq'])
#     print start
# print df_psc_selection.iloc[0]['freq']
#
# samefreq_indices =  abs(df_psc['freq'] -  df_psc_selection.iloc[0]['freq']).argsort()
# for i in samefreq_indices:
#
#print df_psc.loc[418]
# #print df_psc_selection
# control_indices = []
# for name, group in df_psc_selection.groupby('word length'):
#     for name_all,group_all in df_psc.groupby('word length'):
#         if name == name_all:
#             for i, row in group.iterrows():
#                 print abs(group_all['freq'] -  row['freq']).argsort()[:1]
#         print df_psc[df_psc['index value'] == 418]
        # print df_psc[df_psc['index value'] == 418]
        #df_psc_selection.loc[i,'similar word'] = df_psc_bylength.loc[df_psc_bylength.ix[(df_psc_bylength['freq'] - row['freq']).abs().argsort()[:1]].index[0],'words']
        # df_psc_selection.loc[i,'same word'] = df_psc.loc[((df_psc_bylength['freq'] - row['freq']).abs().argsort()[:1][1]),['words']]
        #print df_psc_bylength.ix[(df_psc_bylength['freq'] - row['freq']).abs().argsort()[:1]].index[0]

#print df_psc_selection.loc[0,'words'].any() in df_psc_selection.loc[0,'similar word']
## Select corresponding control words without overlap