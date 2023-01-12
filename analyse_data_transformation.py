__author__ = 'Sam van Leipsig'

import pickle
import numpy as np
from scipy import stats
import pdb
import pandas as pd
import math


def correct_wordskips(df_alldata):
    #Todo can make wordskipped2 with with wordskipindex, use alldata-> 'wordskip index':fixation-1
    ## Adjust/create wordskips
    wordskip2_index = df_alldata.loc[:,'wordskip pass'] == 2
    df_alldata.loc[wordskip2_index,'wordskipped'] =  False ## Remove second pass wordskips
    ## First group by foveal word index, than set foveal word text index-1 with wordskipped2 true: to flag the actual word that is skipped as wordskipped2
    df_alldata_grouped = df_alldata.groupby('foveal word text index', as_index= True)
    wordskiplist = []
    wordskiplist2 = []
    df_alldata['after wordskip'] = df_alldata['wordskipped']
    for name,group in df_alldata_grouped:
       if group.loc[:,'after wordskip'].any() == True:
           wordskiplist.append(name-1)
           wordskiplist2.append(name-2)
    ##TODO apply in df_alldata_grouped
    df_alldata['wordskipped'] = df_alldata['foveal word text index'].isin(wordskiplist)
    df_alldata['before wordskip'] = df_alldata['foveal word text index'].isin(wordskiplist2)

    return df_alldata

def correct_offset(df_alldata):
    for i in range(0,len(df_alldata['Offset'])):
        if df_alldata.loc[i,'word length'] % 2 == 0:
            df_alldata.loc[i,'Offset']-=0.5
    return df_alldata


def make_df_wordactivity(all_data):
    ## Word activities and threshold by word length and frequency #TODO replace with dataframe
    dict_to_plot ={}
    for iteration_data in all_data:
        if iteration_data['regressed']==False:
            dict_to_plot[iteration_data['foveal word']] = [iteration_data['word activities per cycle'],
                                                           iteration_data['fixation word activities'][-4][0],
                                                           iteration_data['fixation word activities'][-4][1],
                                                           iteration_data['fixation word activities'][-4][2],
                                                           iteration_data['fixation word activities'][-4][3],
                                                           iteration_data['fixation word activities'][-4][4],
                                                           iteration_data['word threshold'],len(iteration_data['foveal word']),
                                                          iteration_data['word frequency'],iteration_data['word predictability']]

    # select_word_activity = lambda x: x[-1][3]
    # df_alldata_no_regr2 = df_alldata_no_regr2.reset_index()
    # s_word_activity = df_alldata_no_regr['fixation word activities'].map(select_word_activity)
    df_wordactivity = pd.DataFrame.from_dict(dict_to_plot, orient='index')
    df_wordactivity.columns = ['cycle activity','word excitation','bigram inhibition','betweenword inhibition',
                               'word activity','total activity','word threshold', 'word length','word frequency','word predictability']
    return df_wordactivity

def sequential(x):
    ## Used to detect refixations after a regression
    if (x.index[-1] - x.index[0]) == (len(x)-1):
        return True
    else:
        return False

def make_number_fixations(df_alldata_no_regr):
        ## Select specific fixation duration measures (single, first, second, third)
        ## Important to use sequential in filtering because after a regression a second pass fixation may count as SF
        df_fixation_durations = df_alldata_no_regr.loc[:,['foveal word','fixation duration','foveal word text index','word predictability',
                                                    'refixated','before wordskip','after wordskip','word length','word frequency','saccade distance',
                                                    'relative landing position']]
        df_fixations_sequential = df_fixation_durations.groupby(['foveal word text index']).filter(lambda x: sequential(x))
        df_single_fixation = df_fixations_sequential.groupby(['foveal word text index']).filter(lambda x: len(x)==1)
        first_fixation_grouped =  df_fixations_sequential.groupby(['foveal word text index'])
        first_fixation_selection =  first_fixation_grouped.apply(lambda x: x.index[0]).values
        df_first_fixation = df_fixation_durations.loc[first_fixation_selection,:]

        df_refixations = df_fixation_durations.groupby(['foveal word text index']).filter(lambda x: len(x)>1 and sequential(x))
        df_refixations_grouped = df_refixations.groupby(['foveal word text index'])
        first_fixation_selection_exclusive =  df_refixations_grouped.apply(lambda x: x.index[0]).values
        second_fixation_selection = df_refixations_grouped.apply(lambda x: x.index[1]).values
        print("------------df  fixation locations------------------")
        print(df_fixation_durations)
        df_first_fixation_exclusive = df_fixation_durations.loc[first_fixation_selection_exclusive,:]
        df_second_fixation = df_fixation_durations.loc[second_fixation_selection,:]

        return df_single_fixation, df_first_fixation['fixation duration'], df_second_fixation['fixation duration']


def make_FD_bygroup(df_alldata_no_regr,df_alldata,df_single_fixation,freqbins,predbins):
    ##  Fixation duration grouped by word length and frequency
    df_GD_grpby = df_alldata_no_regr.groupby(['foveal word text index' ]).agg({'fixation duration':np.sum,'word length':np.mean,'word frequency':np.mean,'word predictability':np.mean}).reset_index()
    df_TVT_grpby = df_alldata.groupby(['foveal word text index' ]).agg({'fixation duration':np.sum,'word length':np.mean,'word frequency':np.max,'word predictability':np.mean}).reset_index()

    df_GD_grpby_length = df_GD_grpby.groupby(['word length']).mean()
    df_SF_grpby_length = df_single_fixation.groupby(['word length']).mean()
    df_TVT_grpby_length = df_TVT_grpby.groupby(['word length']).mean()

    word_freq_groups_GD = pd.cut(df_GD_grpby['word frequency'], freqbins)
    word_freq_groups_TVT = pd.cut(df_TVT_grpby['word frequency'], freqbins)
    word_freq_groups_SF = pd.cut(df_single_fixation['word frequency'], freqbins)
    df_GD_grpby_freq = df_GD_grpby.groupby(word_freq_groups_GD).mean()
    df_SF_grpby_freq = df_single_fixation.groupby(word_freq_groups_SF).mean()
    df_TVT_grpby_freq = df_TVT_grpby.groupby(word_freq_groups_TVT).mean()

    word_pred_groups_GD = pd.cut(df_GD_grpby['word predictability'], predbins)
    word_pred_groups_TVT = pd.cut(df_TVT_grpby['word predictability'], predbins)
    word_pred_groups_SF = pd.cut(df_single_fixation['word predictability'], predbins)
    df_GD_grpby_pred = df_GD_grpby.groupby(word_pred_groups_GD).mean()
    df_SF_grpby_pred = df_single_fixation.groupby(word_pred_groups_SF).mean()
    df_TVT_grpby_pred = df_TVT_grpby.groupby(word_pred_groups_TVT).mean()

    FD_bylength_dict = {'SF':df_SF_grpby_length,'GD':df_GD_grpby_length,'TVT':df_TVT_grpby_length}
    FD_byfreq_dict = {'SF':df_SF_grpby_freq,'GD':df_GD_grpby_freq,'TVT':df_TVT_grpby_freq}
    FD_bypred_dict = {'SF':df_SF_grpby_pred,'GD':df_GD_grpby_pred,'TVT':df_TVT_grpby_pred}

    return FD_bylength_dict,FD_byfreq_dict,FD_bypred_dict


def unrecognized_words(unrecognized_words):
    df_unrecognized_words = pd.DataFrame(unrecognized_words)
    df_unrecognized_words.rename(columns={0:'Unrecognized words'}, inplace=True)
    df_unrecognized_words['word length'] = df_unrecognized_words.loc[:,'Unrecognized words'].map(len)
    return df_unrecognized_words


def make_word_measures_bylength(df_alldata):
    word_measures_dict = {}
    df_word_activities = df_alldata['fixation word activities np']
    complete_excitation_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    complete_inhibition_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    complete_activity_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    complete_threshold_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    complete_realactivity_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    complete_decay_matrix = np.zeros((len(df_word_activities),25),dtype=float)
    for i,array in enumerate(df_word_activities):
        complete_excitation_matrix[i,:] = array[:,0]
        complete_inhibition_matrix[i,:] = array[:,2]
        complete_activity_matrix[i,:] = array[:,3]
        complete_threshold_matrix[i,:] = array[:,4]
        complete_realactivity_matrix[i,:] = array[:,5]
        complete_decay_matrix[i,:] = array[:,6]
    df_only_word_activity = pd.DataFrame(complete_activity_matrix, dtype=float)
    df_only_word_activity = df_only_word_activity.loc[:,(df_only_word_activity.any(axis=0)!=0)]
    df_only_word_activity.replace(0.0, np.nan)
    df_only_word_activity = pd.concat([df_only_word_activity,df_alldata['word length']], axis=1, join_axes=[df_only_word_activity.index])


#    word_measures_dict['activity'] = df_only_word_activity.groupby('word length').apply(pd.to_numeric).mean()
    df_only_word_activity = df_only_word_activity.apply(pd.to_numeric)
    with open("test_only_word_activity.pkl","w") as f:
        pickle.dump(df_only_word_activity, f)
    word_measures_dict['activity'] = df_only_word_activity.groupby('word length').mean()

    word_measures_dict['activity'] = word_measures_dict['activity'].apply(pd.to_numeric)


    #df_only_word_activity_grpby_length = df_only_word_activity.groupby('word length').agg(np.mean)

    for i in xrange(len(complete_threshold_matrix[0,:])):
        complete_threshold_matrix[:,i] = complete_threshold_matrix[:,0]
    df_only_word_threshold = pd.DataFrame(complete_threshold_matrix, dtype=float)
    df_only_word_threshold = pd.concat([df_only_word_threshold, df_alldata['word length']], axis=1, join_axes=[df_only_word_threshold.index])
    df_only_word_threshold = df_only_word_threshold.loc[:,(df_only_word_threshold.any(axis=0)!=0)]
    word_measures_dict['threshold'] = df_only_word_threshold.groupby('word length').mean()
    word_measures_dict['threshold'] = word_measures_dict['threshold'].apply(pd.to_numeric)
    #df_only_word_threshold_grpby_length = df_only_word_threshold.groupby('word length').mean()

    df_only_word_excitation = pd.DataFrame(complete_excitation_matrix, dtype=float)
    df_only_word_excitation = pd.concat([df_only_word_excitation,df_alldata['word length']], axis=1, join_axes=[df_only_word_excitation.index])
    df_only_word_excitation = df_only_word_excitation.loc[:,(df_only_word_excitation.any(axis=0)!=0)]
    word_measures_dict['excitation'] = df_only_word_excitation.groupby('word length').mean()
    word_measures_dict['excitation'] = word_measures_dict['excitation'].apply(pd.to_numeric)
    #df_only_word_excitation_grpby_length = df_only_word_excitation.groupby('word length').mean()

    df_only_word_inhibition = pd.DataFrame(complete_inhibition_matrix, dtype=float)
    df_only_word_inhibition = pd.concat([df_only_word_inhibition,df_alldata['word length']], axis=1, join_axes=[df_only_word_inhibition.index])
    df_only_word_inhibition = df_only_word_inhibition.loc[:,(df_only_word_inhibition.any(axis=0)!=0)]
    #word_measures_dict['inhibition'] = df_only_word_inhibition.groupby('word length').mean() # TODO!: Ask why this is commented
    #df_only_word_inhibition_grpby_length = df_only_word_inhibition.groupby('word length').mean()

    df_only_word_realactivity = pd.DataFrame(complete_realactivity_matrix, dtype=float)
    df_only_word_realactivity = pd.concat([df_only_word_realactivity,df_alldata['word length']], axis=1, join_axes=[df_only_word_realactivity.index])
    df_only_word_realactivity = df_only_word_realactivity.loc[:,(df_only_word_realactivity.any(axis=0)!=0)]
    word_measures_dict['realactivation'] = df_only_word_realactivity.groupby('word length').mean()
    word_measures_dict['realactivation'] = word_measures_dict['realactivation'].apply(pd.to_numeric)
    #df_only_word_realactivity_grpby_length = df_only_word_realactivity.groupby('word length').mean()

    df_only_word_decay = pd.DataFrame(complete_decay_matrix, dtype=float)
    df_only_word_decay = pd.concat([df_only_word_decay,df_alldata['word length']], axis=1, join_axes=[df_only_word_decay.index])
    df_only_word_decay = df_only_word_decay.loc[:,(df_only_word_decay.any(axis=0)!=0)]
    word_measures_dict['decay'] = df_only_word_decay.groupby('word length').mean()
    word_measures_dict['decay'] = word_measures_dict['decay'].apply(pd.to_numeric)
    #df_only_word_decay_grpby_length = df_only_word_decay.groupby('word length').mean()

    return word_measures_dict


def regressive_refixation(x):
    if x['regressed'].any() == True:
        if x['regressed'][-1] !=True:
            "REGRESSIVE REFIXATION"
        return True
    return False
