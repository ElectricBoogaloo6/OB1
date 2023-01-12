#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:23:34 2021

@author: nathanvaartjes

This script creates a pickle file that contains the words of the specific task, 
appends their relatives frequencies and predictabilities, as calculated by the SUBTLEX or other relevant resource. 
Also appends 200 most common words of the language to the list 
It puts the generated .dat file in /Data
"""
import chardet
from parameters import return_params
import numpy as np
import pandas as pd
import pickle
import sys

pm = return_params()

task = pm.task_to_run

#get short for language 
lang = pm.short[pm.language]

## lambda functions for converter
comma_to_dot = lambda s: float(s.replace(",","."))
remove_dot = lambda s: s.replace(".","")
decode_ISO= lambda x: x.decode('ISO-8859-1', errors="strict").encode("utf-8")
encode_uft8 = lambda x: x.encode("utf-8",errors="strict")

freqthreshold = 0.15  # 1.5 #NV: why a threshold? For the french lexicon project, this reduces words from 38'000 to 1871. Therefore, almost no overlap
nr_highfreqwords = 500 

def get_words(pm, task): #NV: get_words_task merged with get_words

    if pm.is_experiment: #NV: if experiment, get the relevant freq_pred.txt
        # KM changed to EmbeddedWords_German
        if task == 'EmbeddedWords': #task is special: csv is delimited by ; rather than , Also need to add prime words
            temp = pd.read_csv("Stimuli/"+task+"_stimuli_all_csv.csv", sep=';')
            my_data = list(temp['all'].str.split(' ', expand=True).stack().unique())
            add_data = list(temp['prime'].str.split(' ', expand=True).stack().unique())
            my_data = list(set(my_data + add_data))
            
        else:
            #temp = pd.read_csv(r"C:\Users\Konstantin\Documents\VU_work\OB1_SAM\Stimuli\EmbeddedWords_Nonwords_german_all_csv.csv", sep=';')
            # nonwords lower
            temp = pd.read_csv(r"C:\Users\Konstantin\Documents\VU_work\OB1_SAM\Stimuli\EmbeddedWords_Nonwordslower_german_all_csv.csv", sep=';')
            my_data = list(temp['all'].str.split(' ', expand=True).stack().unique())

     
    else: #NV: structure of PSCAll txt is different, so file reading is different
        my_data = pd.read_csv("Texts/PSCall_freq_pred.txt", usecols=[0], delimiter="\t", encoding='ISO-8859-1') # NV: changed this. Old code resulted in decode error. Actual function is identical

    cleaned_words = [word.replace(".","").lower() for word in my_data]
    return cleaned_words


def create_freq_file(freqthreshold, nr_highfreqwords):
    
    # NV: #!!! This script only builds freq_pred file for task specified as task_to_run in parameters.py, and only for the language specified in parameters pm.language

    # NV: every SUBTLEX or SUBTLEX equivalent has its own column names for the same thing, namely, the Zipf frequency of a word in that language
    
    if pm.language == 'english':
        # NV: get appropriate freq dictionary (SUBTLEX-UK for english, Lexicon Project for french,...). Automatically detects encoding via Chardet and uses the value during import. Due to Chardet, its a bit slow however.
        freqlist_arrays = pd.read_csv("Texts/SUBTLEX_UK.txt", usecols=(0, 1, 5), dtype={'Spelling': np.dtype(str)},
                                      encoding=chardet.detect(open("Texts/SUBTLEX_UK.txt", "rb").read())['encoding'], delimiter="\t")
        freqlist_arrays.sort_values(
            by=['LogFreq(Zipf)'], ascending=False,  inplace=True, ignore_index=True)
        # only keep above threshold words
        freqlist_arrays = freqlist_arrays[freqlist_arrays['LogFreq(Zipf)'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Spelling', 'LogFreq(Zipf)']]
        freq_words = freq_words.rename(columns={'Spelling': 'Word'})

        # NV: is already in zipf, so no tranforming required

    elif pm.language == 'french':
        freqlist_arrays = pd.read_csv("Texts/French_Lexicon_Project.txt",  usecols=(0, 7, 8, 9, 10), dtype={'Spelling': np.dtype(str)},
                                      encoding=chardet.detect(open("Texts/French_Lexicon_Project.txt", "rb").read())['encoding'], delimiter="\t")
        
        freqlist_arrays.sort_values(by=['cfreqmovies'], ascending=False,
                                    inplace=True, ignore_index=True)
        # only keep above threshold words #TODO: figure out filtering (right now, sorts on threshold, but scale is different for every language)
        # freqlist_arrays = freqlist_arrays[freqlist_arrays['cfreqmovies'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word', 'cfreqmovies']]

        # NV: convert to Zipf scale
        freq_words['LogFreq(Zipf)'] = freq_words['cfreqmovies'].apply(lambda x: np.log10(x*1000) if x>0 else 0) # from frequency per million to zipf. Also, replace -inf with 1

        freq_words.drop(columns=['cfreqmovies'], inplace=True)

    elif pm.language == 'german':
        freqlist_arrays = pd.read_csv(r"C:\Users\Konstantin\Documents\VU_work\OB1_SAM\Texts\SUBTLEX_DE.txt", usecols=(0, 1, 3, 4, 5, 9), dtype={'Spelling': np.dtype(str)},
                                      encoding=chardet.detect(open(r"C:\Users\Konstantin\Documents\VU_work\OB1_SAM\Texts\SUBTLEX_DE.txt", "rb").read())['encoding'], delimiter="\t")
        

        freqlist_arrays.sort_values(by=['lgSUBTLEX'], ascending=False,
                                    inplace=True, ignore_index=True)

        # KM Fixing the issue with values not being read correctly
        for (columnName, columnData) in freqlist_arrays.iteritems():
            if columnName == "lgSUBTLEX":
                fix_list = freqlist_arrays['lgSUBTLEX'].tolist()
                fix_list_2 = [i.replace(',', '.') for i in fix_list]
                fix_list_3 = [float(x) for x in fix_list_2]
        
        freqlist_arrays['lgSUBTLEX'] = fix_list_3

        # only keep above threshold words
        freqlist_arrays = freqlist_arrays[freqlist_arrays['lgSUBTLEX'] > freqthreshold]
        # Clean and select frequency words and frequency
        freq_words = freqlist_arrays[['Word', 'lgSUBTLEX']]
        # Making lowercase
        freq_words['Word'] = freq_words['Word'].str.lower()
        #TODO: IS DE in Zipf??
        
    elif pm.language == 'dutch':
        # Read the frequency of the words in the Dutch Lexicon 
        freqlist_arrays = pd.read_csv('Texts/SUBTLEX-NL.txt', encoding = "ISO-8859-1", sep='\t')
        
        freqlist_arrays.sort_values(by=['Zipf'], ascending=False,
                                    inplace=True, ignore_index=True)
        
        freqlist_arrays = freqlist_arrays[freqlist_arrays['Zipf'] > freqthreshold]
        
        freq_words = freqlist_arrays[['Word', 'Zipf']]
        
        
    else:
        raise NotImplementedError(pm.language + " is not implemented yet!")

    #make dict from csv columns word and word frequency
    frequency_words_dict=dict(zip(freq_words[freq_words.columns[0]], freq_words[freq_words.columns[1]]))
    #make array of all words
    frequency_words_np = np.array(freq_words['Word'])
    

    cleaned_words = get_words(pm, task)  # NV: merged get_words with get_words_task
    overlapping_words = list(set(cleaned_words) & set(frequency_words_np)) #changed to set operation rather than np.intersect
    

    # NV: uselful to check out if everything went well: see encoding of cleaned words, see percentage of overlap between dictionary and cleaned words
    print("words in task:\n", cleaned_words)
    print("amount of words in task:", len(cleaned_words))
    print("words in task AND in dictionnary:\n", overlapping_words)
    print("amount of overlapping words", len(overlapping_words))

    # Match PSC/task and freq words and put in dictionary with freq
    file_freq_dict = {}
    for word in overlapping_words:
        file_freq_dict[(word.lower()).strip()] = frequency_words_dict[word.strip()]

    # Put top freq words in dict, can use np.shape(array)[0]):
    for line_number in range(nr_highfreqwords):
        file_freq_dict[((freq_words.iloc[line_number][0]).lower())
                       ] = freq_words.iloc[line_number][1]

    # NV: pickle freq file
    output_file_frequency_map = "Data/" + task + "_frequency_map_"+lang+".dat"
    with open(output_file_frequency_map, "wb") as f:
        pickle.dump(file_freq_dict, f)
    print('frequency file stored in ' + output_file_frequency_map)
    
    # length is important for next function
    return(len(file_freq_dict))


def create_pred_file(task, file_freq_dict_length):
    
    #!!!: this file, as of now, does not create pred files, but imports precomputed pred values 
    # for the Classification and Transposed tasks only. 
       
    #for these tasks, grammar prob is not implemented. So output list with 1's, that will be overwritten later on 
    # (just so the function does not crash)
    if not pm.POS_implemented:
        word_pred_values = np.repeat(1, file_freq_dict_length)
        
    else:
        
        # Add prob to each word in the stimuli according to grammar
        grammar_prob_dt = pd.read_csv('./Texts/POSprob_' + task + '.csv')
        grammar_prob = grammar_prob_dt.values.tolist()
        # Organize the grammar prob into an array of the probability of a certain POS coming before or after
        # with X sentences each and constituted by Y words
        grammar_prob = np.array(grammar_prob)
        if task == 'Sentence':
            word_pred_values = np.reshape(grammar_prob, (2, 400, 4))
        elif task == 'Transposed':
            word_pred_values = np.reshape(grammar_prob, (2, 240, 5))
        elif task == 'Classification':
            word_pred_values = np.reshape(grammar_prob, (2, 200, 3))
            
    
    output_file_pred_map = "Data/"+task+"_predictions_map_"+pm.short[pm.language]+".dat"
    with open(output_file_pred_map, "wb") as f:
        pickle.dump(word_pred_values, f)

        
def get_freq_files(task, pm): #NV: merged the de, fr and en versions. All files have now the task name and language embedded in file name
        
    try: 
        output_word_frequency_map = "Data/"+task+"_frequency_map_"+pm.short[pm.language]+".dat"
        with open (output_word_frequency_map,"rb") as f:
            word_freq_dict = pickle.load(f, encoding="latin1") # For Python3
            
        
        return word_freq_dict
    
    #NV: hint to help troubleshoot
    except:
        raise(FileNotFoundError("Run freq_pred_files.py first!"))
    
def get_pred_files(task, pm): #NV: merged the de, fr and en versions. All files have now the task name and language embedded in file name

    try:
        output_word_predictions_map = "Data/"+task+"_predictions_map_"+pm.short[pm.language]+".dat"
        with open (output_word_predictions_map,"rb") as p:
            word_pred_dict = pickle.load(p, encoding="latin1") # For Python3
        
        return word_pred_dict
    
    #NV: hint to help troubleshoot
    except:
        raise(FileNotFoundError("Run freq_pred_files.py first!"))

def get_freq():
    convert_dict = {0:decode_ISO,1:comma_to_dot, 2:comma_to_dot}
    my_data = np.genfromtxt("Texts/PSCall_freq_pred.txt", names =True, dtype=['U20','f4','f4'], converters = convert_dict, skip_header=0, delimiter="\t")
    return my_data['freq']


def get_pred():
    convert_dict = {0:decode_ISO,1:comma_to_dot, 2:comma_to_dot}
    my_data = pd.read_csv("Texts/PSCall_freq_pred.txt", delimiter="\t", encoding="ISO-8859-1")
    return my_data['pred']
    #predictions_dict = {}
    # for word_ix,word in enumerate(my_data['pred']):
    #     #tempcleanedword = unicode(word['word'].replace(".","").lower())
    #     #predictions_dict[[word_ix,tempcleanedword]] = word['pred']
    #     pred_values_by_index[]
    # return predictions_dict


def get_freq_and_pred():
    convert_dict = {0:decode_ISO,1:comma_to_dot, 2:comma_to_dot}
    # Changed this, old code threw an decode error
    my_data = pd.read_csv("Texts/PSCall_freq_pred.txt", delimiter="\t", encoding="ISO-8859-1") #NV: added encoding parameter. Got encoding via chardet.detect. #TODO: also, remove punctuation, captials, etc?
    #my_data = np.genfromtxt("Texts/PSCall_freq_pred.txt", names =True, encoding="ISO-8859-1",  dtype=['U2','f4','f4'],  skip_header=0, converters = convert_dict, delimiter="\t") #converters = convert_dict,
    predictions_dict = {}
    return my_data

# def get_freq_and_pred_fr(task):  ## NS added for experiment simulations (flanker task and sentence reading task)
#     convert_dict = {0:decode_ISO,1:comma_to_dot, 2:comma_to_dot}
#     # Changed this, old code threw an decode error
#     my_data = pd.read_csv("Texts/"+ task + "_freq_pred.txt",delimiter="\t")
# #    my_data = np.genfromtxt("Texts/PSCall_freq_pred.txt", names =True,encoding="latin-1",  dtype=['U2','f4','f4'], converters = convert_dict, skip_header=0, delimiter="\t")
#     predictions_dict = {}
#     return my_data

def get_freq_and_syntax_pred():
    convert_dict = {0:decode_ISO,1:comma_to_dot, 2:comma_to_dot}
    # Changed this, old code threw an decode error
    my_data = pd.read_csv("Texts/PSCall_freq_pred.txt",delimiter="\t")
    sys.path.append("Data")
    print("Using syntax pred values")
    with open("Data/PSCALLsyntax_probabilites.pkl", "r") as f:
        my_data["pred"] = pickle.load(f)
#    my_data = np.genfromtxt("Texts/PSCall_freq_pred.txt", names =True,encoding="latin-1",  dtype=['U2','f4','f4'], converters = convert_dict, skip_header=0, delimiter="\t")
    predictions_dict = {}
    return my_data



if __name__ == '__main__':
    
    #when running this file directly, create freq & pred files. Other functions are meant to be imported in other modules.
    
    file_freq_dict_length = create_freq_file(freqthreshold, nr_highfreqwords)
    create_pred_file(task, file_freq_dict_length)
