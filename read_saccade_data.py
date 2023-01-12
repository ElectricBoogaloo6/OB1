"""
This file reads the data from the file "Fixation_durations_complete.txt" and converts tihs data to a dataframe.
The data contains information regarding fixation durations and saccade distances. 
The script provides several functions that can be used to analyze the data, such as counting the number of saccades at different distances, 
and calculating the proportion of words that were skipped over in the experiment. "get_sacc_distance()" function counts the number of saccades and gets saccade distances. 
"wordskiplist" contains the proportion of words that were skipped over. 
Finally, "get_saccade_durations()" calculates the duration of the saccade at a certain position, 
"get_saccade_type_probabilities()" function gathers statistical information based on the saccades data, such as refixations (nfp).
"""

__author__ = 'SAM'
import os.path
import pdb
import chardet
import sys
import numpy as np
import pickle
import pandas as pd
from pandas import HDFStore
import matplotlib.pyplot as plt
from parameters import return_params

pm=return_params()


## lambda functions for converter
comma_to_dot = lambda s: float(s.replace(",","."))
remove_dot = lambda s: s.replace(".","")
decode_ISO= lambda x: x.decode('ISO-8859-1', errors="strict").encode("utf-8")
encode_uft8 = lambda x: x.encode("utf-8",errors="strict")


freqbins  = np.arange(0,7,1)
predbins = np.arange(0.1,1.2,0.5)


def write_to_hdf_fixed(df):
    df.to_hdf('Data/Fixation_durations_complete.hdf','complete',mode='w')

def write_to_pkl(df):
    with open("Data/Fixation_durations_"+pm.language+".pkl","w") as f:
        pickle.dump(df, f)

def get_exp_data():
    convert_dict = {column:comma_to_dot for column in [0,1,2,3,4,5]}
    ## todo create own naming, consistent with model naming
    ## See d.all.rd for the description of parameters, from eyetrackR package from Laubrock & Kliegl (2010)
    ## TODO add FFDr to fixation_durations_complete
    my_arrays = np.genfromtxt("Texts/Fixation_durations_complete.txt", dtype= ('i4','i4','i4','i4','i4','i4','i4','i4',
                                                                               'i4','i4','i4','i4','f4','f4','f4','f4',
                                                                               'f4','b','b'),converters= convert_dict , delimiter="\t",names=True)
    saccade_data = pd.DataFrame(my_arrays)
    saccade_data.rename(columns={"nlett1":"word length"}, inplace=True)
    saccade_data['one wordskip'] = pd.Series(np.zeros(len(saccade_data['haveFirstPass']),dtype=bool), index=saccade_data.index)
    for i,value in enumerate(saccade_data.loc[:,'haveFirstPass']):
        if value == 0 and (saccade_data.loc[i-1,'haveFirstPass'] == 1) and (saccade_data.loc[i+1,'haveFirstPass']== 1):
            saccade_data.loc[i,'one wordskip'] = True
    with open("Data/Fixation_durations_"+pm.language+".pkl","w") as f:
        pickle.dump(saccade_data, f)
    write_to_pkl(saccade_data)
    return saccade_data

def read_exp_data():
    filename = "Data/Fixation_durations_"+pm.language+".pkl"
    if os.path.isfile(filename):
        with open(filename,"r") as f:
            return pickle.load(f)
    else:
        return get_exp_data()

def get_sacc_distance():
    saccade_data = read_exp_data()
    saccade_data = saccade_data[saccade_data['ISL']>0]
    sacc_distance_groups = pd.cut(saccade_data['ISL'], np.arange(-5,25,1))
    return saccade_data['ISL'].groupby(sacc_distance_groups).count()/float(len(saccade_data))

def trans_sacc_distance():
    saccade_data = read_exp_data()
    wordskiplist = []
    singlelist = []
    wordskiplist_append = wordskiplist.append
    singlelist_append = singlelist.append
    index_array = saccade_data.index.values
    for i in index_array[0:len(index_array)-1]:
        if (saccade_data.loc[i,'haveFirstPass'] == 1) and (saccade_data.loc[i+1,'haveFirstPass'] ==0):
            wordskiplist_append(i)
        elif (saccade_data.loc[i,'haveFirstPass'] == 1) and (saccade_data.loc[i+1,'haveFirstPass'] ==1):
            singlelist_append(i)
    saccade_data.loc[:,'saccade type'] = np.zeros((len(saccade_data['one wordskip'])),dtype=int)
    saccade_data.loc[singlelist,'saccade type'] = 1
    saccade_data.loc[wordskiplist,'saccade type'] = 2
    saccade_data = saccade_data[saccade_data['ISL']>0]
    # #TODO the regressions are not in d.all dataset, therefore this is not correct
    # #TODO is possible to get this from a dataset, but quite complex
    # regr_index =  saccade_data['OSL'][saccade_data['nsp'] > 0].index
    # saccade_data.loc[regr_index,'ISL']  = saccade_data.loc[regr_index,'ISL'] * -1.
    return saccade_data


def get_sacc_distance_bytype_simple():
    saccdata = read_exp_data()
    saccade_data = trans_sacc_distance()
    sacc_distance_single = saccade_data[(saccade_data['SFD']>0) & (saccade_data['saccade type'] == 1)]
    sacc_distance_wordskip = saccade_data[(saccade_data['SFD']>0) & (saccade_data['saccade type'] == 2)]
    total_words = len(sacc_distance_single) + len(sacc_distance_wordskip)
    saccade_distance_dict = {}
    sacc_distance_groups_single = pd.cut(sacc_distance_single['ISL'], np.arange(-5,25,1))
    saccade_distance_dict['single']=  sacc_distance_single['ISL'].groupby(sacc_distance_groups_single).count()/float(total_words)
    sacc_distance_groups_wordskip = pd.cut(sacc_distance_wordskip['ISL'], np.arange(-5,25,1))
    saccade_distance_dict['wordskipped'] =  sacc_distance_wordskip['ISL'].groupby(sacc_distance_groups_wordskip).count()/float(total_words)
    return saccade_distance_dict


def get_sacc_distance_bytype_hard():
    #TODO only wordskips and single fixations possible
    sacc_distance_regressions = saccade_data[saccade_data['nsp'] > 0]
    sacc_distance_wordskips = saccade_data[(saccade_data['one wordskip'] == 1)]
    sacc_distance_refixations = saccade_data[saccade_data['nfp']>1]
    sacc_distance_single = saccade_data[(saccade_data['SFD']>0)]
    #for wordskips sacc lenght, select the outgoing saccade lenght OSL for the previous word
    #the d.all dataset doesn't have the regression and refixation length!
    # sacc_distance_groups = pd.cut(saccade_data['ISL'], np.arange(-25,25,1))
    # sacc_distance_bytype_dict = {}
    # total_sum = len(sacc_distance_regressions) + len(sacc_distance_wordskips) + len(sacc_distance_refixations) + len(sacc_distance_single)
    # sacc_distance_bytype_dict['regressions'] = sacc_distance_regressions['ISL'].groupby(sacc_distance_groups).count() #/float(total_sum)
    # sacc_distance_bytype_dict['wordskips'] = sacc_distance_wordskips['ISL'].groupby(sacc_distance_groups).count() #/float(total_sum)
    # sacc_distance_bytype_dict['refixations'] = sacc_distance_refixations['ISL'].groupby(sacc_distance_groups).count() #/float(total_sum)
    # sacc_distance_bytype_dict['single'] = sacc_distance_single['ISL'].groupby(sacc_distance_groups).count() #/float(total_sum)
    # for key,value in sacc_distance_bytype_dict.iteritems():
    #     value.plot()


## todo refixation probability & SFD for landing position rel to middle of word
def get_landing_postition():
    groupsize = 20
    saccade_data = read_exp_data()
    saccade_data = saccade_data[saccade_data['ILP'] > 0 ] #have no landing position
    saccade_data['landing position'] = saccade_data['ILP'] * saccade_data['word length']
    saccade_data['relative landing position'] = (saccade_data['landing position']) - (saccade_data['word length']*0.5)
    saccade_data['ISL'] = saccade_data['ISL'].map(round)
    saccade_data['landing position'] = saccade_data['landing position'].map(round)
    saccade_data['relative landing position real'] = saccade_data['relative landing position']
    saccade_data['relative landing position'] = saccade_data['relative landing position'].map(round)

    df_single_fixation = saccade_data[saccade_data['SFD']>0]
    df_single_fixation_filter = df_single_fixation.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
    SF_grpby_rlpos = df_single_fixation_filter.groupby('relative landing position')['SFD'].mean()

    refixations = saccade_data.loc[:,['nfp','relative landing position','word length']]
    refixations['refixated'] = pd.Series(np.zeros(len(refixations),dtype=bool),index= refixations.index)
    row_index_refix = (refixations[refixations['nfp']>1]).index.tolist()
    refixations.loc[row_index_refix, 'refixated'] = True
    #refix_grpby_rlpos =  refixations[refixations['word length']==4]
    refixations_filter = refixations.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
    refixations_grpby_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].sum()
    refixations_groupsizes_rlpos = refixations_filter.groupby(['relative landing position']).size()
    refixations_prob_by_rlpos = refixations_grpby_rlpos/refixations_groupsizes_rlpos
    word_length_bins = np.arange(2,15,4)
    refixations_wl_groups = pd.cut(refixations['word length'], word_length_bins)
    refixations_wl_grouped = refixations.groupby(refixations_wl_groups)

    return saccade_data['relative landing position real'], refixations_wl_grouped, SF_grpby_rlpos
    #return refixations_prob_by_rlpos, SF_grpby_rlpos


def get_FD_bygroup(freqbins,predbins):
    saccade_data = read_exp_data()

    df_single_fixation =  saccade_data[saccade_data['SFD']>0]
    sd_first_fixation_exclusive = saccade_data[saccade_data['FFDp']>0]
    sd_first_fixation = saccade_data[saccade_data['FFDr']>0]
    # df_GD_grpby = saccade_data[saccade_data['GZD']>0]
    # df_TVT_grpby = saccade_data[saccade_data['TVT']>0]
    ## Use nfp and nap to exclude words that only have a gzd and tvt -> erronous data.
    df_GD_grpby = saccade_data[(saccade_data['GZD']>0) & (saccade_data['nfp']>0)]
    df_TVT_grpby = saccade_data[(saccade_data['TVT']>0) & (saccade_data['nap']>0)]
    sd_regressions = saccade_data['TVT'][saccade_data['nsp'] > 0] - saccade_data['GZD'][saccade_data['nsp'] > 0]

    df_GD_grpby_length = df_GD_grpby.groupby(['word length'])['GZD'].aggregate(np.mean)
    df_SF_grpby_length = df_single_fixation.groupby(['word length'])['SFD'].aggregate(np.mean)
    df_TVT_grpby_length = df_TVT_grpby.groupby(['word length'])['TVT'].aggregate(np.mean)

    word_freq_groups_GD = pd.cut(df_GD_grpby['f'], freqbins)
    word_freq_groups_TVT = pd.cut(df_TVT_grpby['f'], freqbins)
    word_freq_groups_SF = pd.cut(df_single_fixation['f'], freqbins)
    df_GD_grpby_freq = df_GD_grpby.groupby(word_freq_groups_GD)['GZD'].aggregate(np.mean)
    df_SF_grpby_freq = df_single_fixation.groupby(word_freq_groups_SF)['SFD'].aggregate(np.mean)
    df_TVT_grpby_freq = df_TVT_grpby.groupby(word_freq_groups_TVT)['TVT'].aggregate(np.mean)

    word_pred_groups_GD = pd.cut(df_GD_grpby['pred'], predbins)
    word_pred_groups_TVT = pd.cut(df_TVT_grpby['pred'], predbins)
    word_pred_groups_SF = pd.cut(df_single_fixation['pred'], predbins)
    df_GD_grpby_pred = df_GD_grpby.groupby(word_pred_groups_GD)['GZD'].aggregate(np.mean)
    df_SF_grpby_pred = df_single_fixation.groupby(word_pred_groups_SF)['SFD'].aggregate(np.mean)
    df_TVT_grpby_pred = df_TVT_grpby.groupby(word_pred_groups_TVT)['TVT'].aggregate(np.mean)

    FD_bylength_dict = {'SF':df_SF_grpby_length,'GD':df_GD_grpby_length,'TVT':df_TVT_grpby_length}
    FD_byfreq_dict = {'SF':df_SF_grpby_freq,'GD':df_GD_grpby_freq,'TVT':df_TVT_grpby_freq}
    FD_bypred_dict = {'SF':df_SF_grpby_pred,'GD':df_GD_grpby_pred,'TVT':df_TVT_grpby_pred}
    return FD_bylength_dict,FD_byfreq_dict,FD_bypred_dict


def get_lagsuccessor(freqbins, predbins):
    file_name = "Data/DF_exp_lag_successor.pkl"
    if not os.path.isfile(file_name):
        saccade_data = read_exp_data()
        saccade_data =  saccade_data[saccade_data['SFD']>0]
        for i in range(1,len(saccade_data)-1):
            ix = saccade_data.index.tolist()[i]
            ixmin1 = saccade_data.index.tolist()[i-1]
            ixplus1 = saccade_data.index.tolist()[i+1]
            saccade_data.loc[ix,'previous freq'] = saccade_data.loc[ixmin1,'f']
            saccade_data.loc[ix,'next freq'] = saccade_data.loc[ixplus1,'f']
            saccade_data.loc[ix,'previous length'] = saccade_data.loc[ixmin1,'word length']
            saccade_data.loc[ix,'next length'] = saccade_data.loc[ixplus1,'word length']
            saccade_data.loc[ix,'previous pred'] = saccade_data.loc[ixmin1,'pred']
            saccade_data.loc[ix,'next pred'] = saccade_data.loc[ixplus1,'pred']

        saccade_data.to_pickle(file_name)
    else:
        saccade_data = pd.io.pickle.read_pickle(file_name)

    df_SF_lagsucc_dict = {}
    df_SF_lagsucc_dict['lag length'] = saccade_data.groupby('previous length')['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ length'] = saccade_data.groupby('next length')['SFD'].agg(np.mean)
    SF_previousfreq_groups = pd.cut(saccade_data['previous freq'], freqbins)
    SF_nextfreq_groups = pd.cut(saccade_data['next freq'], freqbins)
    df_SF_lagsucc_dict['lag freq'] = saccade_data.groupby(SF_previousfreq_groups)['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ freq'] = saccade_data.groupby(SF_nextfreq_groups)['SFD'].agg(np.mean) #.drop('word frequency', 1)
    SF_previouspred_groups = pd.cut(saccade_data['previous pred'], predbins)
    SF_nextpred_groups = pd.cut(saccade_data['next pred'], predbins)
    df_SF_lagsucc_dict['lag pred'] = saccade_data.groupby(SF_previouspred_groups)['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ pred'] = saccade_data.groupby(SF_nextpred_groups)['SFD'].agg(np.mean)

    df_GD_lagsucc_dict = {}
    df_GD_lagsucc_dict['lag length'] = saccade_data.groupby('previous length')['GZD'].agg(np.mean)
    df_GD_lagsucc_dict['succ length'] = saccade_data.groupby('next length')['GZD'].agg(np.mean)
    GD_previousfreq_groups = pd.cut(saccade_data['previous freq'], freqbins)
    GD_nextfreq_groups = pd.cut(saccade_data['next freq'], freqbins)
    df_GD_lagsucc_dict['lag freq'] = saccade_data.groupby(GD_previousfreq_groups)['GZD'].agg(np.mean)
    df_GD_lagsucc_dict['succ freq'] = saccade_data.groupby(GD_nextfreq_groups)['GZD'].agg(np.mean) #.drop('word frequency', 1)
    GD_previouspred_groups = pd.cut(saccade_data['previous pred'], predbins)
    GD_nextpred_groups = pd.cut(saccade_data['next pred'], predbins)
    df_GD_lagsucc_dict['lag pred'] = saccade_data.groupby(GD_previouspred_groups)['GZD'].agg(np.mean)
    df_GD_lagsucc_dict['succ pred'] = saccade_data.groupby(GD_nextpred_groups)['GZD'].agg(np.mean)
    return df_SF_lagsucc_dict, df_GD_lagsucc_dict


#if __name__ == "__main__":
#    freqbins = np.arange(0, 7, 1)  # not sure if correct
#    predbins = np.arange(0.1, 1.2, 0.5)  # not sure if correct
#    df_SF_lagsucc_dict = get_lagsuccessor(freqbins, predbins)
#    print(df_SF_lagsucc_dict)


def get_saccade_durations():
    saccade_data = read_exp_data()
    #def subtract_cols(series):
    sd_single_fixation =  saccade_data['SFD'][saccade_data['SFD']>0]
    sd_first_fixation_exclusive = saccade_data['FFDp'][saccade_data['FFDp']>0]
    sd_first_fixation = saccade_data['FFDr'][saccade_data['FFDr']>0]
    # sd_gaze_duration = saccade_data['GZD'][saccade_data['GZD']>0]
    # sd_tvt = saccade_data['TVT'][saccade_data['TVT']>0]
    ## Use nfp and nap to exclude words that only have a gzd and tvt -> erronous data.
    sd_gaze_duration = saccade_data['GZD'][(saccade_data['GZD']>0) & (saccade_data['nfp']>0)]
    sd_tvt = saccade_data['TVT'][(saccade_data['TVT']>0) & (saccade_data['nap']>0)]
    sd_regressed_after_wordskip = saccade_data[saccade_data['SKP']==1]
    sd_wordskips1 = saccade_data[saccade_data['s1'] == 1] #incoming saccade skipped a word
    sd_wordskips2 = saccade_data[saccade_data['s2'] == 1] #outgoing saccade skipped a word
    #sd_regressions = saccade_data['TVT'][saccade_data['nsp']==1] - saccade_data['GZD'][saccade_data['nsp']==1]
    sd_regressions = saccade_data['TVT'][saccade_data['nsp'] > 0] - saccade_data['GZD'][saccade_data['nsp'] > 0]
    sd_second_fixations = saccade_data['GZD'][saccade_data['nfp']==2] - saccade_data['FFDp'][saccade_data['nfp']==2]
    sd_multiple_fixations = saccade_data['GZD'][saccade_data['nfp']>2] - saccade_data['FFDp'][saccade_data['nfp']>2]
    FD_dict = {'SFD':sd_single_fixation,'FFD':sd_first_fixation,'SecondFD':sd_second_fixations,'GZD':sd_gaze_duration,'TVT':sd_tvt ,'RD':sd_regressions}
    return FD_dict

def get_saccade_type_probabilities():
    saccade_data = read_exp_data()
    # Skipping rate = % of cases that are not fixated in first pass
    # Single fix rate = 1 - (probability skipping & probability refixations)
    # Regressions rate = % of regressions into target word
    # Multiple wordskips (sequentially) are considered wordskips

    #n_regressions = len(saccade_data[saccade_data['nap'] > saccade_data['nfp']])
    n_regressions = len(saccade_data[saccade_data['nsp'] > 0])
    #n_wordskips = len(saccade_data[(saccade_data['haveFirstPass'] == 0)]) #len(saccade_data[saccade_data['s2'] == 1])
    n_wordskips = len(saccade_data[(saccade_data['one wordskip'] == 1)])
    n_refixations = len(saccade_data[saccade_data['nfp']>1])
    # regressed_wordskips = len(saccade_data[(saccade_data['haveFirstPass']==0) & (saccade_data['haveFixation']==1)])
    #n_forward = len(saccade_data[(saccade_data['SFD']>0) & (saccade_data['s2']!=1)]) + len(saccade_data[(saccade_data['FFDp']>0) & (saccade_data['s2']!=1)])

    #n_total = float(len(saccade_data))
    ## Exclude words from sentences that are completely skipped sequentially, except first
    n_total = float(len(saccade_data[(saccade_data['haveFixation']==True) | (saccade_data['one wordskip'] == 1)]))
    mydict= {}
    mydict['regressions'] =  n_regressions/n_total
    mydict['word skips'] = n_wordskips/n_total
    mydict['refixations'] = n_refixations/n_total
    mydict['single fixations'] = 1 - (mydict['refixations'] + mydict['word skips'])

    return mydict

def get_grouped_sacctype_prob(freqbins,predbins):
    df_sacctypes = read_exp_data()
    ## Exclude words from sentences that are completely skipped sequentially, except the first word
    df_sacctypes = df_sacctypes[(df_sacctypes['haveFixation']==True) | (df_sacctypes['one wordskip'] == 1)]
    #df_sacctypes = df_sacctypes.loc[:,['word length','f','pred','nfp','nsp','haveFirstPass']]
    df_sacctypes = df_sacctypes.loc[:,['word length','f','pred','nfp','nsp','one wordskip']]
    df_sacctypes['regressed'] = pd.Series(np.zeros(len(df_sacctypes),dtype=bool),index=df_sacctypes.index)
    row_index_regr = df_sacctypes[df_sacctypes['nsp'] > 0 ].index.tolist()
    df_sacctypes.loc[row_index_regr, 'regressed'] = True
    df_sacctypes['refixated'] = pd.Series(np.zeros(len(df_sacctypes),dtype=bool),index=df_sacctypes.index)
    row_index_refix = (df_sacctypes[df_sacctypes['nfp']>1]).index.tolist()
    df_sacctypes.loc[row_index_refix, 'refixated'] = True
    df_sacctypes['wordskipped'] = pd.Series(np.zeros(len(df_sacctypes),dtype=bool),index=df_sacctypes.index)
    #row_index_skp = (df_sacctypes[(df_sacctypes['haveFirstPass'] == 0)]).index.tolist()
    row_index_skp = (df_sacctypes[(df_sacctypes['one wordskip'] == 1)]).index.tolist()
    df_sacctypes.loc[row_index_skp, 'wordskipped'] = True
    #df_sacctypes = df_sacctypes.drop(['nfp','nsp','haveFirstPass'],1)
    df_sacctypes = df_sacctypes.drop(['nfp','nsp','one wordskip'],1)

    df_sacctypes_grpby_length = df_sacctypes.drop(['f','pred'],1).groupby(['word length']).agg(np.sum)
    groupsizes_length = df_sacctypes.drop(['f','pred'],1).groupby(['word length']).size()

    word_freq_groups_sacc = pd.cut(df_sacctypes['f'], freqbins)
    df_sacctypes_grpby_freq = df_sacctypes.drop(['word length','f','pred'],1).groupby(word_freq_groups_sacc).sum()
    groupsizes_freq = df_sacctypes.drop(['word length','f','pred'],1).groupby(word_freq_groups_sacc).size()

    word_pred_groups_sacc = pd.cut(df_sacctypes['pred'], predbins)
    df_sacctypes_grpby_pred = df_sacctypes.drop(['word length','f','pred'],1).groupby(word_pred_groups_sacc).sum()
    groupsizes_pred = df_sacctypes.drop(['word length','f','pred'],1).groupby(word_pred_groups_sacc).size()

    sacctype_grouped_prob = {}
    sacctype_grouped_prob['length'] = df_sacctypes_grpby_length.div(groupsizes_length, axis=0)
    sacctype_grouped_prob['freq'] = df_sacctypes_grpby_freq.div(groupsizes_freq, axis=0)
    sacctype_grouped_prob['pred'] = df_sacctypes_grpby_pred.div(groupsizes_pred, axis=0)
    return sacctype_grouped_prob

def get_saccade_data_df():
    convert_dict = {0:decode_ISO}
    convert_dict = {column:comma_to_dot for column in [0,1,2,3,4,5]}
    my_arrays = np.genfromtxt("Texts/Fixation_durations_complete.txt", dtype= ('i4','i4','i4','i4','i4','i4','i4','i4','i4','i4'),converters= convert_dict , delimiter="\t",names=True)
    saccade_data = pd.DataFrame(my_arrays)


