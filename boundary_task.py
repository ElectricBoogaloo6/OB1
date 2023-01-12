# __author__ = 'Sam van Leipsig'
from __future__ import division
import pandas as pd
import numpy as np
from boundary_task_functions import *
import os

## Three conditions: Repeated, control, baseline
## Select sentences with where two words of same length are repeated (Stimulus size, 5 words)
## Select a control word from the text that doesn't contain any of the letters of the first word
## Save the word condition in seperate variable
## Apply these sentences during reading
## Create the text
## Select two random filler sentences
## Select sentence with repeat,control or baseline version
## No same condition sequentially, at least one other condition in between.
## do each condition at least twice for each test sentence
## original word position, new word position, condition

filename = "PSC_ALL"
filepath_psc = "PSC/" + filename + ".txt"
BT_filename = 'Data/boundary_task_DF.pkl'

def create_boundary_task(filepath_psc):

    ##PARAMETERS
    length_selection = [4,5]
    max_freq_diff = 1.0
    max_pred = 0.43
    condition_repeats = 3

    ## FIND CONDITION AND CONTROL
    df_psc = create_df_psc(filepath_psc)
    sentence_lenghts_list = parse_sentences(filepath_psc)
    df_psc = add_sentence_numbers(df_psc, sentence_lenghts_list)
    df_psc_selection = select_condition_words(df_psc, length_selection, max_freq_diff, max_pred)
    df_psc = add_control_words(df_psc, df_psc_selection, length_selection, max_freq_diff)
    #df_psc = remove_capitals(df_psc)

    ## MAKE BOUNDARY TASK PSC
    condition_sentence_numbers = np.array(df_psc[df_psc['control indices']!=0]['sentence number'].values,dtype=int)
    # for i in condition_sentence_numbers:
    #     print df_psc[(df_psc['sentence number']==i)]['words'].values, df_psc[(df_psc['sentence number']==i)]['control words'].values
    # print df_psc[(df_psc['sentence number']==114)],condition_sentence_numbers
    filler_sentencenumber_list = create_filler_sentences(df_psc,condition_sentence_numbers)
    df_exp_psc = make_df_boundarytask(condition_sentence_numbers,filler_sentencenumber_list,df_psc,condition_repeats)
    return df_exp_psc


def save_dataframe(df,df_filename):
    df.to_pickle(df_filename)

def read_dataframe(df_filename):
    return pd.read_pickle(df_filename)


create_boundary_task(filepath_psc)
save_dataframe(create_boundary_task(filepath_psc),BT_filename)

