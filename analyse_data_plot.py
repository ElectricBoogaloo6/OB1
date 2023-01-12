__author__ = 'Sam van Leipsig'

import __main__
import os
from time import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import numpy as np
from scipy import stats
import pdb
import nltk
import pandas as pd
import math
import pickle
from freq_pred_files import get_freq_files, get_pred_files
from reading_common import get_stimulus_text_from_file
from parameters import return_params

pm=return_params()

#from pandas.stats.moments import ewma


#TODO make sure df_alldata is not changed in these functions, because it points towards original dataframe

####### DATA ANALYSE

def before_regression(df_alldata):
    error_list2 = df_alldata[(df_alldata['word length']==2) & (df_alldata['regressed']==True) & (df_alldata['wordskipped']==True)].index.tolist()
    error_list = df_alldata[(df_alldata['word length']==2) & (df_alldata['regressed']==True) & (df_alldata['wordskipped']==False)].index.tolist()
    # sprint df_alldata.loc[error_list2[0]-2,:]
    # print df_alldata.loc[error_list2[0]-1,:]


def analyse_fixdur_aroundwordskip(df_alldata,df_unrecognised_words):
    ## Check whether in the fixation step before the wordskip, if the skipped word was recognized
    df_alldata_wordskip =  df_alldata[(df_alldata['wordskipped']==True)]
    wordskiplist = df_alldata_wordskip.index.tolist()
    wordskip_recognized_before = []
    for i in wordskiplist:
        foveal_word_index_wordskipped = df_alldata.loc[i-1,:]['foveal word text index']+1
        recognized_word_positions_beforewordskip = df_alldata.loc[i-1,:]['exact recognized words positions']
        wordskip_recognized_before.append(foveal_word_index_wordskipped in recognized_word_positions_beforewordskip)
    print ("recognized before wordskip:",(np.sum(wordskip_recognized_before)/float(len(wordskip_recognized_before))),float(len(wordskip_recognized_before)))
    #print "% 2 letter words skipped: ",len(df_alldata[(df_alldata['word length']==2) & (df_alldata['wordskipped']==True)])/float(len(df_alldata[df_alldata['word length']==2]) )
    # Check if skipped words are eventually recognized
    wordskip_recognized_after = []
    for word_index in df_alldata[(df_alldata['wordskipped']==True)].loc[:,'foveal word text index']:
        wordskip_recognized_after.append(word_index in df_unrecognised_words.iloc[:,1].values)
    print ("recognized after wordskip:",1-np.sum(wordskip_recognized_after)/float(len(wordskip_recognized_after)))
    #print df_alldata[df_alldata['word length']==2]['word frequency'].mean(), df_alldata[df_alldata['word length']==2]['word predictability'].mean()


def fastvsslow_words(df_single_fixation):
    ## Select the df_words with fast and slow fixation durations
    fast_words = df_single_fixation[df_single_fixation['fixation duration']<200]
    slow_words = df_single_fixation[df_single_fixation['fixation duration']>200]
    print ('Fast words (length,freq):',fast_words['word length'].mean(),fast_words['word frequency'].mean())
    print ('Slow words (length,freq):',slow_words['word length'].mean(),slow_words['word frequency'].mean())



####### DATA PLOTTING

def plot_by_relative_landing_pos(df_single_fixation, df_alldata_grouped_all,exp_refix_prob_by_rlpos,exp_SF_grpby_rlpos):
    ##TODO check if middle of word is calculated in the same way
    groupsize = 20
    single_fixations = df_single_fixation.loc[:,['relative landing position','fixation duration']]
    single_fixations['relative landing position'] = single_fixations['relative landing position'].map(round)
    single_fixations_filter = single_fixations.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
    SF_grpby_rlpos = single_fixations_filter.groupby('relative landing position')['fixation duration'].mean()
    #print (single_fixations_filter.groupby('relative landing position')['fixation duration'].size())
    plt.figure("SF by rel pos")
    plt.title('Single fixation duration by relative landing position')
    plt.ylabel('Fixation duration')
    SF_grpby_rlpos.plot()
    exp_SF_grpby_rlpos.plot()

    # print df_alldata_grouped_all[(df_alldata_grouped_all['refixated']==True) & (df_alldata_grouped_all['word length']>8)]['relative landing position'].mean()
    # print df_alldata_grouped_all[(df_alldata_grouped_all['refixated']==True) & (df_alldata_grouped_all['relative landing position']<0)]['word length'].mean()

    # TODO for groupby object, not sum/mean version
    # second_fixations = df_alldata_grouped_all.groupby(['foveal word text index']).apply(lambda x: x.index[1]).values
    # df_second_fixations_all = df_alldata_grouped_all.loc[second_fixations,:]
    # df_second_fixations = df_second_fixations_all.loc[:,['Offset','refixated','relative landing position','word length']]
    # df_second_fixations['relative landing position'] = df_second_fixations['relative landing position'].map(round)
    # refixations = df_second_fixations

    refixations = df_alldata_grouped_all.loc[:,['Offset','refixated','relative landing position','word length']]
    #refixations['relative landing position'] = refixations['relative landing position'].map(round)
    word_length_bins = np.arange(2,15,4)
    #print word_length_bins
    refixations_wl_groups = pd.cut(refixations['word length'], word_length_bins)
    refixations_wl_grouped = refixations.groupby(refixations_wl_groups)
    plt.figure('Refix by rel pos')
    plt.title('Refixation probability by relative landing position')
    plt.ylabel('Refixation probability')
    for name,groups in refixations_wl_grouped:
        #print groups['word length'].mean()
        refixations_filter = groups.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
        refix_grpby_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].sum()
        refix_groupsizes_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].size()
        #print refix_grpby_rlpos,refix_groupsizes_rlpos
        refix_prob_by_rlpos = refix_grpby_rlpos/refix_groupsizes_rlpos
        refix_prob_by_rlpos.plot()
    for name,groups in exp_refix_prob_by_rlpos:
        #print groups['word length'].mean()
        refixations_filter = groups.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
        refix_grpby_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].sum()
        refix_groupsizes_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].size()
        #print refix_grpby_rlpos,refix_groupsizes_rlpos
        refix_prob_by_rlpos = refix_grpby_rlpos/refix_groupsizes_rlpos
        refix_prob_by_rlpos.plot()
    # refixations_filter = refixations.groupby(['relative landing position']).filter(lambda x: len(x) > groupsize)
    # refix_grpby_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].sum()
    # refix_groupsizes_rlpos = refixations_filter.groupby(['relative landing position'])['refixated'].size()
    # refix_prob_by_rlpos = refix_grpby_rlpos/refix_groupsizes_rlpos
    # refix_prob_by_rlpos.plot()
    #exp_refix_prob_by_rlpos.plot()
    plt.savefig("plots/plot_by_relative_landing_pos.png",dpi=300)

def plot_recognized_cycles(df_alldata):
    print (df_alldata['recognition cycle'].mean())
    df_recognized = df_alldata[df_alldata['recognition cycle']>-1]
    df_grpby_recognized_cycle = df_recognized[['regressed','refixated','forward','wordskipped','recognition cycle']].groupby('recognition cycle').sum()
    fig = plt.figure('Recognized cycles')
    plt.title('Saccade type for cycle of recognition')
    df_grpby_recognized_cycle.astype(int).plot(ax=fig.gca())
    plt.savefig("plots/recognized_cycles.png",dpi=300)

def plot_unrecognizedwords_bytype(df_alldata_grouped_all,df_unrecognized_words):
    unrecognized_index = df_unrecognized_words.iloc[:,1].values
    wordskip_index = df_alldata_grouped_all[(df_alldata_grouped_all['wordskipped']==True)].index
    regressed_index = df_alldata_grouped_all[(df_alldata_grouped_all['regressed']==True)].index
    refixated_index = df_alldata_grouped_all[(df_alldata_grouped_all['refixated']==True)].index
    #single_index = df_alldata_grouped_all[(df_alldata_grouped_all['refixated']==False) & (df_alldata_grouped_all['wordskipped']==True)]
    unrecognized_is_wordskip = set(unrecognized_index).intersection(wordskip_index)
    unrecognized_is_regressed = set(unrecognized_index).intersection(regressed_index)
    unrecognized_is_refixated = set(unrecognized_index).intersection(refixated_index)
    nr_unrecognized_words = len(df_unrecognized_words)
    nr_words = float(len(df_alldata_grouped_all))
    unrecognized_dict = {}
    unrecognized_dict["Wordskips"] = len(unrecognized_is_wordskip)/float(nr_unrecognized_words)
    unrecognized_dict["Regressions"] = len(unrecognized_is_regressed)/float(nr_unrecognized_words)
    unrecognized_dict["Refixations"] = len(unrecognized_is_refixated)/float(nr_unrecognized_words)
    plt.figure("Unrecognized by saccade")
    plt.title("Unrecognized words")
    plt.ylim(0,0.5)
    plt.bar(range(len(unrecognized_dict)),unrecognized_dict.values(),align='center',width=0.5,alpha=0.5)
    plt.xticks(range(len(unrecognized_dict)),unrecognized_dict.keys())
    plt.savefig("plots/plot_unrecognizedwords_bytype.png",dpi=300)



def plot_groupsize_distribution(df_alldata_grouped_all,freqbins,predbins):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))
    plt.setp(axes, ylim=(0,int(len(df_alldata_grouped_all)/2.)))
    fig.canvas.set_window_title('Group size')
    fig.suptitle('Groupsize',fontsize= 16)
    df_alldata_grouped_all.groupby('word length')['foveal word'].count().plot(kind='bar',ax=axes[0,0],alpha=0.4,rot=1,sharex=False)
    alldata_grouped_freqbins = pd.cut(df_alldata_grouped_all['freq'], freqbins)
    df_alldata_grouped_all['freq'].groupby(alldata_grouped_freqbins).count().plot(kind='bar',ax=axes[0,1],alpha=0.4,rot=1,sharex=False)
    alldata_grouped_predbins = pd.cut(df_alldata_grouped_all['pred'], predbins)
    df_alldata_grouped_all['pred'].groupby(alldata_grouped_predbins).count().plot(kind='bar',ax=axes[0,2],alpha=0.4,rot=1,sharex=False)
    wordlen_groupsizes =  df_alldata_grouped_all.groupby('word length').size()
    byfreq_and_length = df_alldata_grouped_all.groupby(alldata_grouped_freqbins)
    axes[1,0].set_ylim(0,1)
    for i,group in byfreq_and_length:
        freq_bylen = group.groupby('word length')['foveal word'].count()/wordlen_groupsizes
        freq_bylen.plot(kind='line',ax=axes[1,0],alpha=0.4,rot=1,sharex=False)
        #group.groupby('word length')['foveal word'].count().plot(kind='line',ax=axes[1,0],alpha=0.4,rot=1,sharex=False)
    axes[1,0].legend(freqbins+2)
    bypred_and_length = df_alldata_grouped_all.groupby(alldata_grouped_predbins)
    axes[1,1].set_ylim(0,1)
    for i,group in bypred_and_length:
        #print group.groupby('word length')['foveal word'].count()
        pred_bylen = group.groupby('word length')['foveal word'].count()/wordlen_groupsizes
        pred_bylen.plot(kind='line',ax=axes[1,1],alpha=0.4,rot=1,sharex=False)
        #group.groupby('word length')['foveal word'].count().plot(kind='line',ax=axes[1,1],alpha=0.4,rot=1,sharex=False)
    axes[1,1].legend(predbins+0.25)
    fig.delaxes(axes[1,2])
    plt.savefig("plots/groupsize_distribution.png",dpi=300)


def plot_freqpred_bylength(df_alldata):
    df_temp = df_alldata.loc[:,['word frequency','word predictability','word length']]
    fig = plt.figure('Freq and Pred by length')
    plt.title('Freq and Pred by length')
    minfreq = min(df_alldata['word frequency'])
    maxfreq = max(df_alldata['word frequency'])
    scalefreq = lambda i: ((i-minfreq)/(maxfreq-minfreq))
    df_temp.loc[:,'word frequency'] = df_temp['word frequency'].map(scalefreq)
    df_temp.groupby('word length').mean().plot(ax = fig.gca())
    plt.savefig("plots/freqpred_bylen.png", dpi=300)


def plot_unrecognizedwords(df_alldata,df_alldata_grouped_all,df_unrecognized_words):
    df_unrecognized_words_by_length_grpsizes = df_alldata_grouped_all.groupby('word length').size()
    df_unrecognized_words_by_length = df_unrecognized_words.groupby('word length')['Unrecognized words'].count()
    df_unrecognized_words_by_length_relative = df_unrecognized_words_by_length/df_unrecognized_words_by_length_grpsizes
    plt.figure('Unrecognized words')
    nr_unrecognized = str(int(len(df_unrecognized_words)/float(max(df_alldata['foveal word text index']))*100.))
    plt.title("Unrecognized words, ("+nr_unrecognized+"% of total)")
    df_unrecognized_words_by_length_relative.plot(kind='bar')
    plt.savefig('plots/unrecognized_words.png', dpi=300)


def plot_attendwidth(df_alldata):
    plt.figure('Attentional width')
    attendwidthbins = np.arange(0,6.5,0.5)
    plt.title("Attentional width")
    df_alldata['attentional width'].hist(bins=attendwidthbins)
    print ("not 5 width skip", len(df_alldata[(df_alldata['attentional width'] != 5) & (df_alldata['after wordskip']==True)]))
    plt.savefig('plots/attentional_width', dpi=300)


def plot_saccdistance(df_alldata_no_regr,exp_saccade_distance):
    ## Saccade distance (preceding)
    plt.figure("Saccade distance")
    plt.title("Saccade distance")
    df_alldata_selection = df_alldata_no_regr[['foveal word text index','saccade distance','after wordskip']]
    df_SF_sacc_distance = df_alldata_selection.groupby(['foveal word text index']).filter(lambda x: len(x)==1)
    total_words = len(df_alldata_no_regr[['saccade distance','foveal word text index']].groupby('foveal word text index').max())
    saccade_distance = df_SF_sacc_distance['saccade distance']
    saccade_distance = saccade_distance.map(abs)
    sacc_distance_groups = pd.cut(saccade_distance, np.arange(-5,25,1))
    sacc_distance = saccade_distance.groupby(sacc_distance_groups).count() / float(total_words)
    sacc_distance.plot(style = 'b')
    exp_saccade_distance.plot(style = 'g--')
    plt.legend(['Sim.','Exp.'])
    plt.savefig('plots/saccade_distance.png', dpi=300)


def sse_saccdistance(df_alldata_no_regr,exp_saccade_distance):
    ## Saccade distance (preceding)
    plt.figure("Saccade distance")
    plt.title("Saccade distance")
    df_alldata_selection = df_alldata_no_regr[['foveal word text index','saccade distance','after wordskip']]
    df_SF_sacc_distance = df_alldata_selection.groupby(['foveal word text index']).filter(lambda x: len(x)==1)
    total_words = len(df_alldata_no_regr[['saccade distance','foveal word text index']].groupby('foveal word text index').max())
    saccade_distance = df_SF_sacc_distance['saccade distance']
    saccade_distance = saccade_distance.map(abs)
    sacc_distance_groups = pd.cut(saccade_distance, np.arange(-5,25,1))
    sacc_distance = saccade_distance.groupby(sacc_distance_groups).count() / float(total_words)
    sacc_distance.plot(style = 'b')
    exp_saccade_distance.plot(style = 'g--')
    plt.legend(['Sim.','Exp.'])
    sse_dist = (((sacc_distance-exp_saccade_distance)*3000)**2).sum()
    plt.xlabel("SSE:"+str(int(sse_dist)))
    t = time()
    plt.savefig(str(t)+'sse_saccdist.png', dpi=300)
    return sse_dist


def plot_saccdistance2(df_alldata_no_regr, exp_saccade_distance):
    ## Saccade distance by saccade type(preceding)
    df_alldata_selection = df_alldata_no_regr[['foveal word text index','saccade distance','after wordskip']]
    df_SF_sacc_distance = df_alldata_selection.groupby(['foveal word text index']).filter(lambda x: len(x)==1)
    df_SF_sacc_distance_single = df_SF_sacc_distance[df_SF_sacc_distance['after wordskip']==0]
    df_SF_sacc_distance_skipped = df_SF_sacc_distance[df_SF_sacc_distance['after wordskip']==1]
    sacc_distance_groups_single = pd.cut(df_SF_sacc_distance_single['saccade distance'], np.arange(-5,25,1))
    sacc_distance_groups_skipped = pd.cut(df_SF_sacc_distance_skipped['saccade distance'], np.arange(-5,25,1))
    sacc_distance_single = df_SF_sacc_distance_single.groupby(sacc_distance_groups_single).count()['saccade distance'] / float(len(df_SF_sacc_distance))
    sacc_distance_skipped = df_SF_sacc_distance_skipped.groupby(sacc_distance_groups_skipped).count()['saccade distance'] / float(len(df_SF_sacc_distance))
    plt.figure("Saccade distance 2")
    plt.title("Saccade distance")
    sacc_distance_single.plot(style = 'b')
    sacc_distance_skipped.plot(style = 'y')
    exp_saccade_distance['single'].plot(style = 'g--')
    exp_saccade_distance['wordskipped'].plot(style = 'y--')
    plt.legend(['Sim. SF','Sim. Skip','Exp. SF','Exp. Skip'])
    plt.savefig('plots/saccade_distance2.png', dpi=300)


def plot_offset(df_alldata,exp_landing_positions):
    plt.figure('Offset')
    #df_alldata[df_alldata['refixated']==False]['Offset'].hist(bins=10)
    plt.title('Landing positions distribution')
    plt.xlabel('Initial landing position')
    plt.ylabel('Fixation probability')
    df_alldata_firstpass = df_alldata[(df_alldata['refixated']==False) | df_alldata['regressed']==False] #/max(df_alldata['foveal word text index'])
    # df_alldata_firstpass['relative landing position'].map(round).hist(normed=True,alpha=0.2, color = 'b')
    # exp_landing_positions.map(round).hist(normed=True,alpha=0.2, color = 'g')
    #ewma(exp_landing_positions, span=35).plot(style='k')
    df_alldata_firstpass['relative landing position'].astype(int).plot(kind = 'kde',color = 'b',bw_method=0.4)
    exp_landing_positions.plot(kind = 'kde',color = 'g',bw_method=0.4)
    plt.xlim(-7,7)
    plt.legend(['Sim.','Exp.'])
    print(df_alldata_firstpass['relative landing position'].mean(),exp_landing_positions.mean())
    print("Refixated (Offset, saccade error):",df_alldata[df_alldata['refixated']==False]['Offset'].mean(), df_alldata[df_alldata['refixated']==False]['saccade error'].mean())
    plt.savefig("plots/offset.png",dpi=300)


#TODO plot previous saccdistance by
def plot_saccadedistance_bytype(df_alldata):
    plt.figure('Saccade distance by type')
    #TODO adjust bar width accordingly
    plt.title('Saccade distance by type')
    df_alldata[df_alldata['after wordskip']==True].groupby('saccade distance')['fixation duration'].count().plot()
    df_alldata[df_alldata['regressed']==True].groupby('saccade distance')['fixation duration'].count().plot()
    df_alldata[df_alldata['refixated']==True].groupby('saccade distance')['fixation duration'].count().plot()
    df_alldata[df_alldata['forward']==True].groupby('saccade distance')['fixation duration'].count().plot()
    plt.legend(['after wordskip','regressed','refixation','forward'])
    plt.savefig('plots/saccade_distance_by_type.png', dpi=300)
    # SF_saccdistance_groups = pd.cut(df_single_fixation['saccade distance'], distancebins)
    # df_SF_bydistance = df_single_fixation.groupby(SF_saccdistance_groups)['fixation duration'].mean()
    # plt.figure(18)
    # plt.title("Single fixation duration by saccade distance")
    # df_SF_bydistance.plot()


def plot_saccerror(df_alldata):
    plt.figure('Saccade error')
    plt.title('Saccade error distribution')
    plt.ylabel('Fixation probability')
    #df_alldata.rename(columns={'saccade_error': 'saccade error'},inplace=True)
    #print df_alldata['saccade error'].mean()
    saccErr_bins  = np.arange(math.floor(np.min(df_alldata['saccade error'])),math.ceil(np.max(df_alldata['saccade error'])),0.5)
    saccErr_groups = pd.cut(df_alldata['saccade error'], saccErr_bins)
    saccade_errors = df_alldata['saccade error'].groupby(saccErr_groups).count()/max(df_alldata['foveal word text index'])
    saccade_errors.plot()
    plt.savefig('plots/saccade_error.png', dpi=300)


def plot_saccerror_type(df_alldata):
    ## Saccade type caused by error
    plt.figure('Saccade error by type')
    plt.title('Saccade type by error')
    df_alldata['saccade_type_by_error'].hist(bins=8)
    xrange = int(max(df_alldata['saccade_type_by_error'].unique()))
    plt.xticks(range(xrange+1),['No error','refixated','forward','after wordskip'], size='large', alpha=0.5, ha='center')
    plt.savefig('plots/saccade_type_by_error.png', dpi=300)


def plot_refix_types(df_alldata):
    ## Refixation types
    plt.figure('Refixation types')
    plt.title('Refixations types')
    df_alldata[df_alldata['refixation type']>0]['refixation type'].hist(alpha=0.5,bins=3)
    plt.xticks((1,2,3),['Not recognized','Activity','Saccade error'],alpha=0.5)
    plt.savefig('plots/refixation_types.png', dpi=300)


def plot_saccadetype_probabilities(df_alldata_grouped_all,exp_sacc_dict):
    ## Number of each saccade type, probability per word (not saccade)
    n_words = float(len(df_alldata_grouped_all))
    p_wordskips = sum(df_alldata_grouped_all['wordskipped'])/n_words
    p_regressions = sum(df_alldata_grouped_all['regressed'])/n_words
    p_refixated = sum(df_alldata_grouped_all['refixated'])/n_words
    p_single = 1 - (p_wordskips+p_refixated)
    exp_sacc_dict = [exp_sacc_dict['regressions'],exp_sacc_dict['refixations'],exp_sacc_dict['single fixations'],exp_sacc_dict['word skips']]
    exp_data_rayner = [0.1,0.15,0.55,0.30]
    model_data = [p_regressions,p_refixated,p_single,p_wordskips]
    plt.figure('Saccade type probability')
    plt.title("Saccade type probability")
    barwidth = 0.2
    x = np.arange(0,float(len(exp_sacc_dict)))
    plt.bar(x, exp_data_rayner,width = barwidth, alpha = 0.2, color='r')
    plt.bar(x+barwidth,exp_sacc_dict,width = barwidth, alpha = 0.4, color = 'g')
    plt.bar(x+(2.*barwidth),model_data,width = barwidth, alpha = 0.4, color = 'b')
    plt.legend(['Rayner','Exp.','Simulation'])
    plt.ylabel('Fixation probability')
    plt.xticks(x+barwidth, ['regressed','refixated','single','wordskip'], size='medium')
    plt.savefig('plots/saccade_type_probabilities.png', dpi=300)

##todo make freq/word frequency consistent
def plot_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins):
    maketrue = lambda col: 1 if np.sum(col)>0 else 0
    # funcdict = {'columnname':function}
    # df_sacctypes1 =  df_alldata_grouped[['regressed','refixated','wordskipped']].agg(maketrue)
    df_sacctypes = df_alldata_grouped_all.loc[:,['regressed','refixated','wordskipped','word length','pred','freq']]
    print("wordskip==regressed:", len(df_sacctypes[(df_sacctypes['regressed']==True) & (df_sacctypes['wordskipped']==True)]))
    print("wordskip<regressed:",len(df_sacctypes[(df_sacctypes['regressed']==True) & (df_sacctypes['wordskipped']==False)]))
    print("wordskip>regressed:",len(df_sacctypes[(df_sacctypes['regressed']==False) & (df_sacctypes['wordskipped']==True)]))
    df_sacctypes_grpby_length = df_sacctypes.drop(['freq','pred'],1).groupby(['word length']).sum()
    groupsizes_length = df_sacctypes.drop(['freq','pred'],1).groupby(['word length']).size()
    word_freq_groups_sacc = pd.cut(df_sacctypes['freq'], freqbins)
    df_sacctypes_grpby_freq = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_freq_groups_sacc).sum()
    groupsizes_freq = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_freq_groups_sacc).size()
    word_pred_groups_sacc = pd.cut(df_sacctypes['pred'], predbins)
    df_sacctypes_grpby_pred = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_pred_groups_sacc).sum()
    groupsizes_pred = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_pred_groups_sacc).size()
    print(groupsizes_length)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5),sharey=False)
    fig.canvas.set_window_title('Grouped saccade type prob')
    fig.suptitle('Saccade type probability',fontsize= 20)
    df_sacctypes_grpby_length_prob = df_sacctypes_grpby_length.div(groupsizes_length, axis=0)
    df_sacctypes_grpby_freq_prob = df_sacctypes_grpby_freq.div(groupsizes_freq, axis=0)
    df_sacctypes_grpby_pred_prob = df_sacctypes_grpby_pred.div(groupsizes_pred, axis=0)
    axes[0].set_ylabel('fixation probability')
    axes[0].set_ylim([0,1])
    df_sacctypes_grpby_length_prob.plot(ax=axes[0], style = ['g','r','b'])
    exp_sacctype_grpby_prob_dict['length'].plot(ax=axes[0], style = ['g--','r--','b--'])
    df_sacctypes_grpby_freq_prob.plot(ax=axes[1], style = ['g','r','b'])
    axes[0].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})
    exp_sacctype_grpby_prob_dict['freq'].plot(ax=axes[1], style = ['g--','r--','b--'])
    axes[1].set_xlabel('Log frequency')
    axes[1].set_xticks([0,1,2])
    axes[1].set_xticklabels(['Low','Medium',"High"])
    axes[1].set_ylim([0,0.6])
    axes[1].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})
    df_sacctypes_grpby_pred_prob.plot(ax=axes[2], style = ['g','r','b'])
    exp_sacctype_grpby_prob_dict['pred'].plot(ax=axes[2],style = ['g--','r--','b--'])
    axes[2].set_xticks([0,1,2])
    axes[2].set_xlabel('Predictability')
    axes[2].set_xticklabels(['Low','Med','High'])
    axes[2].set_ylim([0,0.5])
    axes[2].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})
    plt.savefig('plots/saccade_types_grouped.png', dpi=300)


def sse_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins):
    maketrue = lambda col: 1 if np.sum(col)>0 else 0
    # funcdict = {'columnname':function}
    # df_sacctypes1 =  df_alldata_grouped[['regressed','refixated','wordskipped']].agg(maketrue)
    df_sacctypes = df_alldata_grouped_all.loc[:,['regressed','refixated','wordskipped','word length','pred','freq']]
    print("wordskip==regressed:", len(df_sacctypes[(df_sacctypes['regressed']==True) & (df_sacctypes['wordskipped']==True)]))
    print("wordskip<regressed:",len(df_sacctypes[(df_sacctypes['regressed']==True) & (df_sacctypes['wordskipped']==False)]))
    print("wordskip>regressed:",len(df_sacctypes[(df_sacctypes['regressed']==False) & (df_sacctypes['wordskipped']==True)]))
    df_sacctypes_grpby_length = df_sacctypes.drop(['freq','pred'],1).groupby(['word length']).sum()
    groupsizes_length = df_sacctypes.drop(['freq','pred'],1).groupby(['word length']).size()
    word_freq_groups_sacc = pd.cut(df_sacctypes['freq'], freqbins)
    df_sacctypes_grpby_freq = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_freq_groups_sacc).sum()
    groupsizes_freq = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_freq_groups_sacc).size()
    word_pred_groups_sacc = pd.cut(df_sacctypes['pred'], predbins)
    df_sacctypes_grpby_pred = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_pred_groups_sacc).sum()
    groupsizes_pred = df_sacctypes.drop(['word length','freq','pred'],1).groupby(word_pred_groups_sacc).size()
    print(groupsizes_length)

    df_sacctypes_grpby_length_prob = df_sacctypes_grpby_length.div(groupsizes_length, axis=0)
    df_sacctypes_grpby_freq_prob = df_sacctypes_grpby_freq.div(groupsizes_freq, axis=0)
    df_sacctypes_grpby_pred_prob = df_sacctypes_grpby_pred.div(groupsizes_pred, axis=0)

    # SSE for length
    result_length = df_sacctypes_grpby_length_prob - exp_sacctype_grpby_prob_dict['length']
    sse_length = ((result_length * 1000)**2).sum().sum()
    # SSE for freq
    result_freq = df_sacctypes_grpby_freq_prob - exp_sacctype_grpby_prob_dict['freq']
    sse_freq = ((result_freq * 1000)**2).sum().sum()
    # SSE for pred
    result_pred = df_sacctypes_grpby_pred_prob - exp_sacctype_grpby_prob_dict['pred']
    sse_pred = ((result_pred * 1000)**2).sum().sum()
    # Total SSE
    sse_total = sse_length + sse_freq + sse_pred

    t = time()

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5),sharey=True)
    fig.canvas.set_window_title('Grouped saccade type prob')
    fig.suptitle('Saccade type probability',fontsize= 20)
    axes[0].set_ylabel('fixation probability')
    axes[0].set_ylim([0,1])

    axes[0].set_xlabel('word length\n'+str(sse_length))

    df_sacctypes_grpby_length_prob.plot(ax=axes[0], style = ['g','r','b'])
    exp_sacctype_grpby_prob_dict['length'].plot(ax=axes[0], style = ['g--','r--','b--'])
    df_sacctypes_grpby_freq_prob.plot(ax=axes[1], style = ['g','r','b'])
    axes[0].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})
    exp_sacctype_grpby_prob_dict['freq'].plot(ax=axes[1], style = ['g--','r--','b--'])

    axes[1].set_xlabel('Log frequency\n'+str(sse_freq))

    axes[1].set_xticks([0,1,2])
    axes[1].set_xticklabels(['Low','Medium',"High"])
    axes[1].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})
    df_sacctypes_grpby_pred_prob.plot(ax=axes[2], style = ['r','g','b'])
    exp_sacctype_grpby_prob_dict['pred'].plot(ax=axes[2],style = ['g--','r--','b--'])
    axes[2].set_xticks([0,1,2])

    axes[2].set_xlabel('Predictability\n'+str(sse_pred))

    axes[2].set_xticklabels(['Low','Med','High'])
    axes[2].legend(['Sim. Regressions','Sim. Refixations','Sim. Wordskips','Exp. Regressions','Exp. Refixations','Exp. Wordskips'], loc=2,prop={'size':12})

    plt.savefig(str(t)+'_pred_sse.png', dpi=300)
    plt.close()

    print("SSE saccade_type_prob:")
    print("length:"+str(sse_length))
    print("freq:"+str(sse_freq))
    print("pred:"+str(sse_pred))
    print("total:"+str(sse_total))

    if pm.sacc_type_objective == "length":
        return sse_length
    if pm.sacc_type_objective == "freq":
        return sse_freq
    if pm.sacc_type_objective == "pred":
        return sse_pred
    return sse_total


##TODO find out why lagsucc SF duration differ so much
def plot_lagsuccessor(df_alldata_no_regr,df_single_fixation,freqbins,predbins,dict1, SF_OR_GD, wordlength_limit):
    dict1 = dict1
    df_lagsuccessor = df_alldata_no_regr.groupby('foveal word text index')['word length', 'word frequency','word predictability'].median()
    for i in range(1,len(df_lagsuccessor)-1):
        ix = df_lagsuccessor.index.tolist()[i]
        ixmin1 = df_lagsuccessor.index.tolist()[i-1]
        ixplus1 = df_lagsuccessor.index.tolist()[i+1]
        df_lagsuccessor.loc[ix,'previous freq'] = df_lagsuccessor.loc[ixmin1,'word frequency']
        df_lagsuccessor.loc[ix,'next freq'] = df_lagsuccessor.loc[ixplus1,'word frequency']
        df_lagsuccessor.loc[ix,'previous length'] = df_lagsuccessor.loc[ixmin1,'word length']
        df_lagsuccessor.loc[ix,'next length'] = df_lagsuccessor.loc[ixplus1,'word length']
        df_lagsuccessor.loc[ix,'previous pred'] = df_lagsuccessor.loc[ixmin1,'word predictability']
        df_lagsuccessor.loc[ix,'next pred'] = df_lagsuccessor.loc[ixplus1,'word predictability']

    df_SF_lagsuccessor = df_single_fixation['fixation duration']
    #df_SF_lagsuccessor = df_gaze_duration['fixation duration']
    df_SF_lagsuccessor = pd.concat([df_SF_lagsuccessor,df_lagsuccessor], axis='columns', join_axes=[df_SF_lagsuccessor.index])
    df_SF_lagsuccessor = df_SF_lagsuccessor.ix[1:]
    df_SF_lagsuccessor= df_SF_lagsuccessor[df_SF_lagsuccessor['word length'] < wordlength_limit]
    df_SF_lag_length = df_SF_lagsuccessor.groupby('previous length')['fixation duration'].mean()
    df_SF_successor_length = df_SF_lagsuccessor.groupby('next length')['fixation duration'].mean()
    SF_previousfreq_groups = pd.cut(df_SF_lagsuccessor['previous freq'], freqbins)
    SF_nextfreq_groups = pd.cut(df_SF_lagsuccessor['next freq'], freqbins)
    df_SF_lag_freq = df_SF_lagsuccessor.groupby(SF_previousfreq_groups)['fixation duration'].mean()
    df_SF_successor_freq = df_SF_lagsuccessor.groupby(SF_nextfreq_groups)['fixation duration'].mean() #.drop('word frequency', 1)
    SF_previouspred_groups = pd.cut(df_SF_lagsuccessor['previous pred'], predbins)
    SF_nextpred_groups = pd.cut(df_SF_lagsuccessor['next pred'], predbins)
    df_SF_lag_pred = df_SF_lagsuccessor.groupby(SF_previouspred_groups)['fixation duration'].mean()
    df_SF_successor_pred = df_SF_lagsuccessor.groupby(SF_nextpred_groups)['fixation duration'].mean()

    fig2, axes = plt.subplots(nrows=2, ncols=2,figsize=(10,6),sharey=False)
    fig2.canvas.set_window_title('Lag successor')
    if SF_OR_GD == "GD":
        fixdur_measure = "gaze duration"
        plt.setp(axes, ylim=(200,260))

        fig2.suptitle('Lag and successor effects on foveal ' + fixdur_measure, fontsize= 16)
        df_SF_lag_freq.plot(ax=axes[0,0],title ='Lag, word N-1', kind='line',sharex=False)
        dict1['lag freq'].plot(ax=axes[0,0],title ='Lag, word N-1', style = 'g--', kind='line',sharex=False)
        axes[0,0].set_xlabel('Log frequency')
        axes[0,0].set_xticks([0,1,2])
        axes[0,0].set_xticklabels(['Low','Medium',"High"])
        axes[0,0].set_ylabel(fixdur_measure)
        axes[0,0].legend(['Sim.','Exp.'])
        df_SF_successor_freq.plot(ax=axes[0,1],title='Successor, word N+1', kind='line',sharex=False)
        dict1['succ freq'].plot(ax=axes[0,1],title='Successor, word N+1', style = 'g--', kind='line',sharex=False)
        axes[0,1].set_xlabel('Log frequency')
        axes[0,1].set_xticks([0,1,2])
        axes[0,1].set_xticklabels(['Low','Medium',"High"])
        df_SF_lag_pred.plot(ax=axes[1,0], kind='line',sharex=False)
        dict1['lag pred'].plot(ax=axes[1,0], style = 'g--', kind='line',sharex=False)
        axes[1,0].set_xticks([0,1,2])
        axes[1,0].set_xlabel('Predictability')
        axes[1,0].set_xticklabels(['Low','medium','High'])
        axes[1,0].set_ylabel(fixdur_measure)
        df_SF_successor_pred.plot(ax=axes[1,1], kind='line',sharex=False)
        dict1['succ pred'].plot(ax=axes[1,1], style = 'g--', kind='line',sharex=False)
        axes[1,1].set_xticks([0,1,2])
        axes[1,1].set_xlabel('Predictability')
        axes[1,1].set_xticklabels(['Low','Med','High'])

    elif SF_OR_GD == "SF":
        fixdur_measure = "single fixation duration"
        plt.setp(axes, ylim=(190,240))
        fig2.suptitle('Lag and successor effects on foveal ' + fixdur_measure, fontsize= 16)
        df_SF_lag_freq.plot(ax=axes[0,0],title ='Lag, word N-1', kind='line',sharex=False)
        dict1['lag freq'].plot(ax=axes[0,0],title ='Lag, word N-1', style = 'g--', kind='line',sharex=False)
        axes[0,0].set_xlabel('Log frequency')
        axes[0,0].set_xticks([0,1,2])
        axes[0,0].set_xticklabels(['Low','Medium',"High"])
        axes[0,0].set_ylabel(fixdur_measure)
        axes[0,0].legend(['Sim.','Exp.'])
        df_SF_successor_freq.plot(ax=axes[0,1],title='Successor, word N+1', kind='line',sharex=False)
        dict1['succ freq'].plot(ax=axes[0,1],title='Successor, word N+1', style = 'g--', kind='line',sharex=False)
        axes[0,1].set_xlabel('Log frequency')
        axes[0,1].set_xticks([0,1,2])
        axes[0,1].set_xticklabels(['Low','Medium',"High"])
        df_SF_lag_pred.plot(ax=axes[1,0], kind='line',sharex=False)
        dict1['lag pred'].plot(ax=axes[1,0], style = 'g--', kind='line',sharex=False)
        axes[1,0].set_xticks([0,1,2])
        axes[1,0].set_xlabel('Predictability')
        axes[1,0].set_xticklabels(['Low','medium','High'])
        axes[1,0].set_ylabel(fixdur_measure)
        df_SF_successor_pred.plot(ax=axes[1,1], kind='line',sharex=False)
        dict1['succ pred'].plot(ax=axes[1,1], style = 'g--', kind='line',sharex=False)
        axes[1,1].set_xticks([0,1,2])
        axes[1,1].set_xlabel('Predictability')
        axes[1,1].set_xticklabels(['Low','Med','High'])

    # df_SF_lag_length.plot(ax=axes2[2,1], kind='line',title ='Lag, word N-1',sharex=False)
    # exp_SF_lagsucc_dict['lag length'].plot(ax=axes2[2,1], style = 'g--', kind='line',title ='Lag, word N-1',sharex=False)
    # df_SF_successor_length.plot(ax=axes2[2,2], kind='line',title='Successor, word N+1',sharex=False)
    # exp_SF_lagsucc_dict['succ length'].plot(ax=axes2[2,2], style = 'g--', kind='line',title='Successor, word N+1',sharex=False)

    plt.savefig('plots/Lag_and_successor_effects.png', dpi=300)


def plot_wordactivity_atshift_bylength(df_alldata_no_regr):
    """Does not incorporate words that are skipped"""
    df_wordactivity_shift = df_alldata_no_regr.groupby('foveal word text index')['between word inhibition','bigram inhibition','word excitation','word activity','word threshold', 'word length'].median()
    #df_wordactivity_shift['netinput'] = df_wordactivity_shift['word excitation'] - df_wordactivity_shift['bigram inhibition']
    #df_wordactivity_shift[df_wordactivity_shift['word length']==2][['bigram inhibition','word excitation', 'netinput']].plot()
    df_wordactivity_shift = df_wordactivity_shift.groupby('word length').mean()
    fig = plt.figure('Word activity at shift')
    df_wordactivity_shift.plot(ax = fig.gca(), kind='line')
    plt.title('Word activities vs threshold at shift cycle')
    plt.savefig('plots/Word_activity_atshift.png', dpi=300)


def word_activity_threshold(df_wordactivity):
    df_wordactivity_sort = df_wordactivity.sort('word threshold',ascending = True)
    plt.figure('Word activities vs threshold')
    plt.title('Word activities vs threshold')
    xaxis = np.arange(0,len(df_wordactivity),float(1))
    plt.plot(xaxis,df_wordactivity_sort['word activity'],alpha=0.5,color = 'g') #bar_width,align='center'
    plt.plot(xaxis, df_wordactivity_sort['word threshold'],alpha=0.5,color ='b')


def plot_wordactivity_grouped(df_wordactivity,df_single_fixation,freqbins,predbins):
    df_wordactivity_grpby_length = df_wordactivity.groupby(['word length']).mean().drop(['word frequency', 'word predictability'], 1)
    word_freq_groups_act = pd.cut(df_wordactivity['word frequency'], freqbins)
    df_wordactivity_grpby_freq = df_wordactivity.groupby(word_freq_groups_act).mean().drop(['word length','word frequency','word predictability'], 1)
    word_pred_groups_act = pd.cut(df_wordactivity['word predictability'], predbins)
    df_wordactivity_grpby_pred = df_wordactivity.groupby(word_pred_groups_act).mean().drop(['word length','word frequency','word predictability'], 1)
    df_single_fixation2 = df_single_fixation.set_index('foveal word')
    df_single_fixation2['word activity'] = df_wordactivity['word activity']
    df_single_fixation2['word threshold'] = df_wordactivity['word threshold']
    fd_bins = np.arange(125,275,25)
    df_SF_duration_groups = pd.cut(df_single_fixation2['fixation duration'], fd_bins)
    df_wordactivity_grpby_FD = df_single_fixation2.groupby(df_SF_duration_groups)['word activity', 'word threshold'].mean()

    fig1, axes1 = plt.subplots(nrows=1, ncols=4)
    fig1.canvas.set_window_title('Grouped word activity')
    df_wordactivity_grpby_length.plot(ax=axes1[0],title='Mean word activity for word length')
    df_wordactivity_grpby_freq.plot(ax=axes1[1],title='Mean word activity for word frequency')
    df_wordactivity_grpby_pred.plot(ax=axes1[2],title='Mean word activity for word predictability')
    df_wordactivity_grpby_FD.plot(ax=axes1[3],title='Mean word activity for SF duration')
    plt.savefig('plots/word_activity_grouped.png', dpi=300)


def plot_FD_bygroup(mod_FD_bylength_dict,mod_FD_byfreq_dict,mod_FD_bypred_dict,exp_FD_bylength_dict,exp_FD_byfreq_dict,exp_FD_bypred_dict):
    plt.figure('Fixation durations grouped',figsize = (10,5))
    ax = plt.subplot(221)
    styles = ['b']
    ax.set_title("Fixation duration by word length")
    mod_FD_bylength_dict['SF']['fixation duration'].plot(style=['g'])
    exp_FD_bylength_dict['SF'].plot(style=['g--'])
    mod_FD_bylength_dict['GD']['fixation duration'].plot(style=['r'])
    exp_FD_bylength_dict['GD'].plot(style=['r--'])
    mod_FD_bylength_dict['TVT']['fixation duration'].plot(style=['b'])
    exp_FD_bylength_dict['TVT'].plot(style=['b--'])
    ax.set_ylim(150,500)
    ax.set_ylabel('Fixation duration')
    plt.legend(['SF','SFexp','GD','GDexp','TVT','TVTexp'], loc=2)

    ax = plt.subplot(222)
    ax.set_title("Fixation duration by word frequency")
    mod_FD_byfreq_dict['SF']['fixation duration'].plot(style=['g'])
    exp_FD_byfreq_dict['SF'].plot(style=['g--'])
    mod_FD_byfreq_dict['GD']['fixation duration'].plot(style=['r'])
    exp_FD_byfreq_dict['GD'].plot(style=['r--'])
    mod_FD_byfreq_dict['TVT']['fixation duration'].plot(style=['b'])
    exp_FD_byfreq_dict['TVT'].plot(style=['b--'])
    ax.set_xlabel("Log frequency")
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Low','Medium','High'])
    ax.set_ylim(150,500)
    plt.legend(['SF','SFexp','GD','GDexp','TVT','TVTexp'])
    plt.savefig('plots/Fixation_durations_grouped.png', dpi=300)

    # ax = plt.subplot(223)
    # ax.set_title("Fixation duration by word predictability")
    # mod_FD_bypred_dict['SF']['fixation duration'].plot(style=['g'])
    # exp_FD_bypred_dict['SF'].plot(style=['g--'])
    # mod_FD_bypred_dict['GD']['fixation duration'].plot(style=['r'])
    # exp_FD_bypred_dict['GD'].plot(style=['r--'])
    # mod_FD_bypred_dict['TVT']['fixation duration'].plot(style=['b'])
    # exp_FD_bypred_dict['TVT'].plot(style=['b--'])
    # ax.set_xlabel("Log frequency")
    # ax.set_xticks([0,1,2])
    # ax.set_xticklabels(['Low','Medium','High'])
    # ax.set_ylim(150,500)
    # plt.legend(['SF','SFexp','GD','GDexp','TVT','TVTexp'])
    # plt.savefig('plots/Fixation_durations_grouped.png')



def plot_FD_hists(total_viewing_time,gaze_durations,df_single_fixations,first_fixation,second_fixation,df_FD_only_regr,exp_FD_dict):
    sizediff = len(exp_FD_dict['TVT'])/float(len(total_viewing_time['fixation duration']))
    n_bins = 30
    n_bins_ext = int(n_bins*2)
    plt.figure('Fixation durations',figsize = (12,10))

    myxticks = range(0,500,50)
    # ax = plt.subplot(321)
    # ax.set_title("Total viewing time")
    # plt.hist(total_viewing_time['fixation duration'],bins=n_bins,alpha=0.5, normed = 0)
    # exp_FD_dict['TVT'].hist(bins=n_bins_ext,alpha=0.5, color = 'g', normed = 0)

    band_width = 0.31
    ax = plt.subplot(321)
    ax.set_title("Total viewing time")
    ax.set_xlim([0, 500])
    #plt.setp(ax.get_xticklabels(), visible=True)
    total_viewing_time['fixation duration'].plot(alpha=0.5, kind = 'density',sharex=False, bw_method=band_width)
    exp_FD_dict['TVT'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False, bw_method=band_width)
    plt.legend(['Simulation','Experiment'])


    # ax = plt.subplot(322)
    # ax.set_title("Gaze duration")
    # plt.hist(gaze_durations['fixation duration'],bins=n_bins,alpha=0.5, normed=True)
    # ax = plt.subplot(322)
    # ax.set_title("Gaze duration")
    # exp_FD_dict['GZD'].hist(bins=n_bins_ext,alpha=0.5, normed=True, color = 'g')

    ax = plt.subplot(322)
    ax.set_title("Gaze duration")
    ax.set_xlim([0, 500])
    gaze_durations['fixation duration'].plot(alpha=0.5, kind = 'density',sharex=False, bw_method=band_width)
    exp_FD_dict['GZD'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False, bw_method=band_width)
    print(gaze_durations['fixation duration'].describe())
    print(exp_FD_dict['GZD'].describe())


    # ax = plt.subplot(323)
    # ax.set_title("Single fixation durations")
    # df_single_fixations['fixation duration'].hist(bins=n_bins,normed=True,alpha=0.5)
    # ax = plt.subplot(323)
    # exp_FD_dict['SFD'].hist(bins=n_bins,normed=True,alpha=0.5,color = 'g')

    ax = plt.subplot(323)
    ax.set_title("Single fixation durations")
    ax.set_xlim([0, 500])
    df_single_fixations['fixation duration'].plot(alpha=0.5, kind = 'density',sharex=False, bw_method=band_width)
    exp_FD_dict['SFD'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False, bw_method=band_width)
    print(df_single_fixations['fixation duration'].describe())
    print(exp_FD_dict['SFD'].describe())
    print(df_single_fixations['fixation duration'].mean())
    print(exp_FD_dict['SFD'][(exp_FD_dict['SFD']<350) & (exp_FD_dict['SFD']>125)].mean())

    # ax = plt.subplot(324)
    # ax.set_title("First fixation durations")
    # plt.hist(first_fixation.values,bins=n_bins,alpha=0.5,normed=True)
    # ax = plt.subplot(324)
    # ax.set_title("First fixation durations")
    # exp_FD_dict['FFD'].hist(bins=n_bins,alpha=0.5,normed=True,color = 'g')
    # #print np.mean(first_fixation),np.mean(exp_FD_dict['FFD'])

    ax = plt.subplot(324)
    ax.set_title("First fixation durations")
    ax.set_xlim([0, 500])
    first_fixation.plot(alpha=0.5, kind = 'density',sharex=False, bw_method=band_width)
    exp_FD_dict['FFD'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False, bw_method=band_width)

    # ax = plt.subplot(325)
    # ax.set_title("Second fixation durations")
    # if len(second_fixation)>5:
    #     plt.hist(second_fixation.values,alpha=0.5,bins=n_bins,normed=True,color = 'b')
    # ax = plt.subplot(325)
    # ax.set_title("Second fixation durations")
    # exp_FD_dict['SecondFD'].hist(alpha=0.5,bins=n_bins,normed=True,color = 'g')

    ax = plt.subplot(325)
    ax.set_title("Second fixation durations")
    ax.set_xlim([0, 500])
    if len(second_fixation)>5:
        second_fixation.plot(alpha=0.5, kind = 'density',sharex=False,bw_method=band_width)
    exp_FD_dict['SecondFD'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False,bw_method=band_width)
    ax.set_xlabel('Fixation duration')

    # ax = plt.subplot(325)
    # #ax.set_title("Third fixation duration")
    # plt.hist(third_fixation,alpha=0.5,bins=n_bins,normed=True)

    # ax = plt.subplot(326)
    # ax.set_title("Regressions")
    # df_FD_only_regr.hist(bins=n_bins,alpha=0.5,normed=True)
    # ax = plt.subplot(326)
    # ax.set_title("Regressions")
    # exp_FD_dict['RD'].hist(bins=n_bins,alpha=0.5,normed=True,color='g')

    ax = plt.subplot(326)
    ax.set_title("Regressions")
    ax.set_xlim([0, 500])
    df_FD_only_regr.plot(alpha=0.5, kind = 'density',sharex=False,bw_method=band_width)
    exp_FD_dict['RD'].plot(alpha=0.5, kind = 'density',color = 'g',sharex=False,bw_method=band_width)
    ax.set_xlabel('Fixation duration')

    plt.savefig('plots/Fixation_durations.png', dpi=300)


def is_similar_word_length(word1,word2):
    return abs(len(word1)-len(word2)) < (0.25* max(len(word1),len(word2)))

def plot_word_similarity(df_alldata_grouped_all,max_wordlength):
    lexicon_file_name = "Data/Lexicon.dat"
    with open (lexicon_file_name,"r") as lex:
        lexicon = pickle.load(lex)
        df_lexicon = pd.DataFrame(lexicon)
        df_lexicon.rename(columns={0: 'foveal word'},inplace=True)
        df_lexicon['word length'] = df_lexicon['foveal word'].map(len)
        #df_length_count = df_alldata_grouped_all.groupby('word length')['foveal word'].count()
        df_length_count = df_lexicon.groupby('word length')['foveal word'].count()
        is_similar_dict = {}
        for x in df_length_count.index:
            templist = []
            for i in df_length_count.index:
                if is_similar_word_length(x*'A',i*'A'):
                    templist.append(df_length_count.loc[x])
                else:
                    templist.append(0)
            is_similar_dict[x] = templist
        df_is_similar = pd.DataFrame(is_similar_dict,index=df_length_count.index)
        plt.figure("Number is_similar word")
        plt.title("Number similar words by length")
        plt.ylabel("Word proportion")
        plt.xlabel("Word length")
        plt.plot(df_is_similar.index,df_is_similar.sum(axis=1)/len(lexicon)) #important to sum columns for each row/index
    plt.savefig("plots/word_similarity.png", dpi=300)


def plot_activity_percycle_bylenght(df_only_word_activity_grpby_length,df_only_word_threshold_grpby_length):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    fig.canvas.set_window_title('Word activation per cycle')
    fig.suptitle('Word activation per cycle',fontsize= 16)
    plt.setp(axes, xlim=(0,15),ylim=(0,0.15))
    axes[0].set_ylabel('Activation')
    for ax in axes:
        ax.set_xlabel('Cycles')
    df_only_word_activity_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes[0], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_threshold_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes[0], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_activity_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes[1], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_threshold_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes[1], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_activity_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes[2], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_threshold_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes[2], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    plt.savefig("plots/plot_activity_percycle_bylenght.png", dpi=300)


def plot_exc_inh_percycle_bylength(df_only_word_excitation_grpby_length,df_only_word_inhibition_grpby_length):
    fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    fig2.canvas.set_window_title('Word measures per cycle')
    fig2.suptitle('Word excitation and inhibition per cycle',fontsize= 16)
    plt.setp(axes2, ylim=(0.0,0.03))
    axes2[0].set_ylabel('Strength')
    for ax in axes2:
        ax.set_xlabel('Cycles')
    df_only_word_excitation_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes2[0], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_inhibition_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes2[0], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_excitation_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes2[1], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_inhibition_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes2[1], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_excitation_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes2[2], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_inhibition_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes2[2], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    plt.savefig("plots/exc_inh_bylength.png",dpi=300)

def plot_realactivity_decay_bylength(df_only_word_realactivity_grpby_length, df_only_word_decay_grpby_length):
    fig3, axes3 = plt.subplots(nrows=1, ncols=3, figsize=(20,7))
    fig3.canvas.set_window_title('Word activation function per cycle')
    fig3.suptitle('Word activation and decay per cycle',fontsize= 16)
    plt.setp(axes3, ylim=(-0.01,0.02))
    axes3[0].set_ylabel('Strength')
    for ax in axes3:
        ax.set_xlabel('Cycles')
    df_only_word_realactivity_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes3[0], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_decay_grpby_length.loc[[2,3,4,5],:].T.plot(ax=axes3[0], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_realactivity_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes3[1], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_decay_grpby_length.loc[[6,7,8,9],:].T.plot(ax=axes3[1], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    df_only_word_realactivity_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes3[2], style = ['b','g','r','y'],xlim=(0,15))
    df_only_word_decay_grpby_length.loc[[10,11,12,13],:].T.plot(ax=axes3[2], style = ['b--','g--','r--','y--'], legend=False,xlim=(0,15))
    plt.savefig("plots/plot_realactivity_decay_bylength.png",dpi=300)


def plot_overlapmatrix_by(df_alldata_grouped_all,freqbins):
    #TODO save overlap*activeword for each word of lexicon/individual words
    output_inhibition_matrix = 'Data/Inhibition_matrix.dat'
    with open(output_inhibition_matrix,"r") as k:
        overlap_matrix = pickle.load(k)
        df_overlap_matrix = pd.DataFrame(overlap_matrix)
        df_sum_overlap  = df_overlap_matrix.sum(axis=1)
    df_overlap = pd.concat([df_sum_overlap,df_alldata_grouped_all[['word length','freq','pred']]], axis=1, join_axes=[df_alldata_grouped_all.index])
    df_overlap.rename(columns={0:'total overlap'}, inplace=True)
    df_overlap_grpby_length = df_overlap.groupby(['word length'])[['total overlap']].mean()
    overlap_freq_groups = pd.cut(df_overlap['freq'], freqbins)
    df_overlap_grpby_freq = df_overlap.groupby(overlap_freq_groups)[['total overlap']].mean()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    fig.canvas.set_window_title('Word-to-word inhibition')
    fig.suptitle('Total overlap for length and freq',fontsize= 16)
    axes[0].set_ylabel('Total overlap')
    df_overlap_grpby_length.plot(ax=axes[0])
    df_overlap_grpby_freq.plot(ax=axes[1])

def plot_runtime(stimulus, N_ngrams_lexicon, lexicon_activewords_np, lexicon_word_inhibition_np, 
                 word_input_np, lexicon_word_activity_np, lexicon_thresholds_np, lexicon,
                 words_to_annotate):
    
    ### Plot 1: plot 4 metrics per Ngram length
    
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f'{stimulus = }')
    
    fig.set_size_inches(10, 10)
    
    #get highest active words to mark them
    highest_words = np.argpartition(word_input_np, -4)[-4:]
    words_to_annotate.extend(highest_words)
    
    #make list of affix indexes
    affixes=np.array(['word' if word.startswith('_') else 'affix' for word in lexicon])[lexicon_activewords_np == True]
    
    #loop over plots
    for (x_coord, y_coord, x_type, y_type, title) in [(0,0,N_ngrams_lexicon, word_input_np, 'word activation per length '),
                                               (0,1,N_ngrams_lexicon, lexicon_word_inhibition_np, 'word inhibition per length '),
                                               (1,0,N_ngrams_lexicon, lexicon_word_activity_np, 'total word activity per length'),
                                               (1,1,N_ngrams_lexicon, lexicon_thresholds_np, 'word thresholds per length')]:
        
        #plot relevant metric (only plot active words) on relevant axes
        sns.stripplot(ax=axes[x_coord][y_coord], x=np.array(x_type)[lexicon_activewords_np == True],
                      y=y_type[lexicon_activewords_np == True], hue = affixes)
        axes[x_coord][y_coord].set_title(title)
        
        #annotate most active words
        for word in words_to_annotate:
            x = x_type[word]
            y = y_type[word]
            axes[x_coord][y_coord].plot(x, y, 'ro')
            axes[x_coord][y_coord].text(x, y, f'{lexicon[word]}')

    
    
    plt.show()
    
    ### Plot 2: plot 4 metrics per distance from stimulus
    
    fig, axes = plt.subplots(2, 2)
    fig.suptitle(f'{stimulus = }')
    fig.set_size_inches(10, 10)
    
    #x axis: distance from stimulus
    x_type = np.array([nltk.edit_distance(stimulus.replace('_', ''), x.replace('_', '')) for x in lexicon])
    
    for (x_coord, y_coord, x_type, y_type, title) in \
        [(0,0, x_type, word_input_np, 'word activation per distance '),
        (0,1,x_type, lexicon_word_inhibition_np, 'word inhibition per distance '),
        (1,0, x_type, lexicon_word_activity_np, 'total word activity per distance'),
        (1,1, x_type, lexicon_thresholds_np, 'word thresholds per distance')]:
    
        sns.stripplot(ax=axes[x_coord][y_coord], x=np.array(x_type)[lexicon_activewords_np == True], 
                      y=y_type[lexicon_activewords_np == True], hue = affixes)
        axes[x_coord][y_coord].set_title(title)
        
        for word in words_to_annotate:
            x = x_type[word]
            y = y_type[word]
            axes[x_coord][y_coord].plot(x, y, 'ro')
            axes[x_coord][y_coord].text(x, y, f'{lexicon[word]}')

    plt.show()
        
    
def plot_inhib_spectrum(lexicon, lexicon_activewords_np, inhib_spectrum1, inhib_spectrum2,
                        index_num1, index_num2, inhib_spectrum1_indices, inhib_spectrum2_indices, 
                        cur_cycle):

    if any(lexicon_activewords_np):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(15, 7)
        
        x_coord=0
        
        for (IS, index_num, indices) in [(inhib_spectrum1,index_num1, inhib_spectrum1_indices) , (inhib_spectrum2, index_num2, inhib_spectrum2_indices)]:
        
            sns.stripplot(ax=axes[x_coord],
                          x=np.array(lexicon)[lexicon_activewords_np == True][indices][:10], 
                          y = IS[:10])
            axes[x_coord].set_title(lexicon[index_num])
            x_coord+=1
            
        plt.suptitle(f'{cur_cycle = }')
        plt.show()
    else:
        pass

                