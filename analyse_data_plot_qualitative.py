__author__ = 'Sam van Leipsig'

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy import stats
import pdb

def get_neighbor_data(df_alldata_no_regr):
    ## Fix dur by neighborhoodsize
    ## UPDATED this to the clean version
    output_neighbors = 'Data/individualwords_neighbors_cleaned.dat'
    with open( output_neighbors,"r") as b:
        individual_words_neighbors = pickle.load(b)
    df_alldata_to_group2 = df_alldata_no_regr[['fixation duration','word length','foveal word text index','word frequency']]
    df_alldata_grouped_neighbors = df_alldata_to_group2.groupby('foveal word text index', as_index= True).agg(
        {'fixation duration':np.sum,'word length':np.mean,'word frequency':np.mean})
    df_alldata_grouped_neighbors = pd.concat([df_alldata_grouped_neighbors, individual_words_neighbors[
        ['neighborhoodsize','neighborhoodsize highfreq','neighborhoodsize lowfreq']] ],axis=1,join_axes=[df_alldata_grouped_neighbors.index])
    return df_alldata_grouped_neighbors


def plot_GD_byneighbors(df_alldata_grouped_neighbors, neighborbins):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5),sharey=False)
    fig.canvas.set_window_title('Neighborhood size effects')
    fig.suptitle('Neighborhood Size effect',fontsize= 20)
    word_neighbor_bins_all = pd.cut(df_alldata_grouped_neighbors['neighborhoodsize highfreq'], neighborbins)
    df_alldata_grouped_neighbors[['fixation duration','neighborhoodsize highfreq']].groupby(word_neighbor_bins_all).mean()['fixation duration'].plot(ax=axes[0])
    axes[0].set_title('Gaze duration by high frequency neighbors')
    axes[0].set_xlabel('High frequency neighbors')
    axes[0].set_ylabel('Gaze duration')
    #axes[0].set_xticklabels(['low','','med','','high'])
    #axes[0].set_ylim(200,250)
    df_alldata_freqselection = df_alldata_grouped_neighbors[df_alldata_grouped_neighbors['word frequency']>4.0]
    word_neighbor_bins = pd.cut(df_alldata_freqselection['neighborhoodsize highfreq'], neighborbins)
    df_alldata_freqselection[['fixation duration','neighborhoodsize highfreq']].groupby(word_neighbor_bins).mean()['fixation duration'].plot(ax=axes[1])
    axes[1].set_title('Gaze duration by high frequency neighbors (Control)')
    axes[1].set_ylabel('Gaze duration high frequency words')
    axes[1].set_xlabel('High frequency neighbors')
    #axes[1].set_xticklabels(['low','','med','','high'])
    axes[1].set_ylim(200,250)

def correct_recognition(x):
    if type(x['actual recognized words']) == list:
        if x['actual recognized words']:
            if x['foveal word'] == x['actual recognized words'][0]:
                return True
            else:
                return False
        else:
                return False
    else:
        return False


def preprocess_boundary_task(df_single_first_fixation, df_alldata_no_regr,df_individual_words):
    ## Boundary task
    df_single_first_fixation['correct recognition'] = np.zeros(len(df_single_first_fixation['foveal word']),dtype=bool)
    df_single_first_fixation_select = df_single_first_fixation[['boundary task condition','fixation duration','refixated','foveal word','actual recognized words']]
    boundary_task_POF_indices = df_single_first_fixation_select[df_single_first_fixation_select['boundary task condition']!=0].index
    df_SF_boundary_task_POF = df_single_first_fixation_select.ix[boundary_task_POF_indices]
    boundary_task_preview_indices  = df_SF_boundary_task_POF.index + 1

    df_SF_boundary_task_preview = df_single_first_fixation_select.ix[boundary_task_preview_indices]
    df_SF_boundary_task_preview['boundary task condition']  = df_SF_boundary_task_POF['boundary task condition'].values
    df_SF_boundary_task_POF['correct recognition'] = df_SF_boundary_task_POF.apply(lambda x: correct_recognition(x), axis=1)
    df_SF_boundary_task_preview['correct recognition'] = df_SF_boundary_task_preview.apply(lambda x: correct_recognition(x), axis=1)

    df_alldata_no_regr['correct recognition'] = np.zeros(len(df_alldata_no_regr['foveal word']),dtype=bool)
    df_GD_boundary_task = df_alldata_no_regr #[['foveal word text index','fixation duration', 'boundary task condition','refixated']]
    df_GD_boundary_task_POF = df_GD_boundary_task[df_GD_boundary_task['boundary task condition']!=0]
    df_GD_boundary_task_POF = df_GD_boundary_task_POF.groupby(['foveal word text index' ]).agg({
        'fixation duration':np.sum,'boundary task condition':np.max,'refixated':np.sum,'refixation type':np.sum,'actual recognized words':np.sum,
        'between word inhibition':np.sum,'stimulus competition atshift':np.sum,'word activity':np.sum,'recognition cycle':np.sum})
    df_GD_boundary_task = df_GD_boundary_task.groupby(['foveal word text index' ]).agg({
        'fixation duration':np.sum,'boundary task condition':np.max,'refixated':np.sum,'refixation type':np.sum,'actual recognized words':np.sum,
        'between word inhibition':np.sum,'stimulus competition atshift':np.sum,'word activity':np.sum,'recognition cycle':np.sum})
    boundary_task_preview_indices_GD  = df_GD_boundary_task_POF.index + 1
    df_GD_boundary_task_preview = df_GD_boundary_task.ix[boundary_task_preview_indices_GD]
    df_GD_boundary_task_preview['boundary task condition'] = df_GD_boundary_task_POF['boundary task condition'].values
    df_GD_boundary_task_preview = df_GD_boundary_task_preview[df_GD_boundary_task_preview['boundary task condition']>0]

    df_GD_boundary_task_preview['foveal word'] = df_individual_words.ix[ df_GD_boundary_task_preview.index]['foveal word']
    df_GD_boundary_task_POF['foveal word'] = df_individual_words.ix[df_GD_boundary_task_POF.index]['foveal word']
    df_GD_boundary_task_preview['correct recognition'] = df_GD_boundary_task_preview.apply(lambda x: correct_recognition(x), axis=1)
    df_GD_boundary_task_POF['correct recognition'] = df_GD_boundary_task_POF.apply(lambda x: correct_recognition(x), axis=1)



    ## POF
    print("POF")
    print (df_SF_boundary_task_POF.groupby('boundary task condition').mean())
    print (df_GD_boundary_task_POF.groupby('boundary task condition').mean())
    ## Preview
    print("Preview")
    print (df_SF_boundary_task_preview.groupby('boundary task condition').mean())
    print (df_GD_boundary_task_preview.groupby('boundary task condition').mean())

    pdb.set_trace()
    return df_SF_boundary_task_POF, df_GD_boundary_task_POF

def plot_boundary_task(df_SF_boundary_task, df_GD_boundary_task):
    SF_groupedby_boundarytask = df_SF_boundary_task.groupby('boundary task condition')
    GD_groupedby_boundarytask = df_GD_boundary_task.groupby('boundary task condition')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,7))
    fig.canvas.set_window_title('Orthographical overlap')
    df_SF_boundary_task.boxplot(ax= axes[0], by = 'boundary task condition', column = 'fixation duration')
    axes[0].scatter([1,2,3],SF_groupedby_boundarytask.mean()['fixation duration'])
    axes[0].set_ylabel('Fixation duration')
    axes[0].set_title('First fixation')
    axes[0].set_ylim(100,500)
    df_GD_boundary_task.boxplot(ax= axes[1],by = 'boundary task condition', column = 'fixation duration')
    axes[1].scatter([1,2,3],GD_groupedby_boundarytask.mean()['fixation duration'])
    axes[1].set_title('Gaze duration')
    axes[1].set_ylim(100,500)
    plt.setp(axes, xticks=[1,2,3], xticklabels=['Baseline','Repeat','Control'])
    plt.suptitle("Orthographical parafoveal on foveal effect",fontsize=15)


 ## todo convert to function and check for different freq values etc.
 ## single fixation duration grouped by word competition in complete lexicon
#output_inhibition_matrix = 'Data/Inhibition_matrix.dat'
#with open(output_inhibition_matrix,"r") as b:
#    individual_words_competition = pickle.load(b)
#individual_words_competition = pd.DataFrame(individual_words_competition)
#df_alldata_grouped_all = pd.concat([df_alldata_grouped_all,individual_words_competition], axis=1, join_axes=[df_alldata_grouped_all.index])
#df_alldata_grouped_all.rename(columns={0:'sum competition'}, inplace=True)
#def devide_by_wordlength(series):
#    return (series['sum competition']/series['word length'])
#df_alldata_grouped_all['relative competition'] = df_alldata_grouped_all.apply(devide_by_wordlength,axis=1)
##df_SF_wordindex = df_SF.set_index('foveal word text index2')
#df_SF_wordindex = df_single_first_fixation.set_index('foveal word text index2')
#df_alldata_grouped_all_SF = pd.concat([df_SF_wordindex['fixation duration'],df_alldata_grouped_all], axis=1, join_axes=[df_alldata_grouped_all.index])
#df_alldata_grouped_all_SF_bylen = df_alldata_grouped_all_SF.groupby('word length')
#fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,7))
#fig.canvas.set_window_title('Single Fix by lexicon competition')
#fig.suptitle('SF by lexicon competition',fontsize= 16)
#plt.ylabel('Single fixation duration')
#legend_list = np.zeros((14),dtype=int)
#axistoplot = 0
#for i,group in df_alldata_grouped_all_SF_bylen:
#    if len(group)>25:
#        numberaxis = i-2
#        legend_list[i] = i
#        # if i>7:
#        #     axistoplot=1
#        if numberaxis % 4 == 0:
#            axistoplot += 1
#        numberaxis = numberaxis - ((axistoplot-1)*float(4))
#        group.loc[:,['sum competition']] = (group['sum competition'] - group['sum competition'].min()) / (group['sum competition'].max() - group['sum competition'].min())
#        competition_bins = np.arange(0,1.1,0.25)
#        temp_grouping = pd.cut(group['sum competition'], competition_bins)
#        # competition_bins = np.arange(0,max(df_alldata_grouped_all['sum competition']),250)
#        # temp_grouping = pd.cut(group['sum competition'], competition_bins)
#        grouped_filtered = group.groupby(temp_grouping).filter(lambda x: len(x) > 5)
#        grouped_to_plot = grouped_filtered.groupby(temp_grouping).mean()['fixation duration']
#        #grouped_to_plot = group.groupby(temp_grouping).mean()['fixation duration']
#        grouped_to_plot.plot(ax=axes[axistoplot-1,numberaxis],sharex=False)
#        #group.plot(x='sum competition',y='fixation duration',ax=axes[axistoplot-1,numberaxis])
#        axes[axistoplot-1,numberaxis].legend([i])
#        #axes[axistoplot-1,numberaxis].set_xticklabels(competition_bins)
#        #axes[axistoplot-1,numberaxis].set_xticklabels(['low','med','high'])
#        axes[axistoplot-1,numberaxis].set_ylim([180,230])
#        axes[axistoplot-1,numberaxis].set_ylabel('SF duration')
#        axes[axistoplot-1,numberaxis].set_xlabel('Normalized competition')
#fig = plt.figure('Relative overlap')
#lex_competition_bins = np.arange(0,375,25)
#lex_competition_grouping = pd.cut(df_alldata_grouped_all_SF['relative competition'], lex_competition_bins)
#df_alldata_grouped_all_SF_filtered = df_alldata_grouped_all_SF.groupby(lex_competition_grouping).filter(lambda x: len(x) > 10)
#df_alldata_grouped_all_bycomp = df_alldata_grouped_all_SF_filtered.groupby(lex_competition_grouping).mean()
#df_alldata_grouped_all_bycomp['fixation duration'].plot(ax=fig.gca())
#plt.xlabel('Relative overlap')
#plt.ylabel('Single fixation duration')
#df_SF_selection = df_single_first_fixation.loc[:,['stimulus competition','stimulus competition atshift','word length','foveal word text index','fixation duration']]
#stimulus_competition = df_SF_selection.groupby('foveal word text index').mean().reset_index()
#stimulus_competition.fillna(0)
## fig = plt.figure('Relative stimulus competition')
## stimulus_competition['relative stimulus competition'] = stimulus_competition['stimulus competition']/stimulus_competition['word length']
## overlap_bins_rel = np.arange(0,round(stimulus_competition['relative stimulus competition'].max()+1),2.5)
## overlap_grouping_rel = pd.cut(stimulus_competition['relative stimulus competition'], overlap_bins_rel)
## stimulus_competition_grpby_competition = stimulus_competition.groupby(overlap_grouping_rel).mean()
## stimulus_competition_grpby_competition = stimulus_competition.groupby('relative stimulus competition').mean()
## stimulus_competition_grpby_competition['fixation duration'].plot(ax=fig.gca())
## plt.ylabel('SF duration')
#fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10,5),sharex=False)
#fig.canvas.set_window_title('SF by stimulus competition')
#fig.suptitle('Single fixation duration by parafoveal overlap',fontsize= 25)
#axistoplot = 0
#stimulus_competition_grpby_length = stimulus_competition.groupby('word length')
#overlap_bins = np.arange(0,round(stimulus_competition['stimulus competition'].max()+0.5),2.5)
#overlap_grouping = pd.cut(stimulus_competition['stimulus competition'], overlap_bins)
#legend_list = np.zeros((14),dtype=int)
#for i,group in stimulus_competition_grpby_length:
#    if len(group['stimulus competition'])>50:
#        numberaxis = i-2
#        legend_list[i] = i
#        if numberaxis % 4 == 0:
#            axistoplot += 1
#        numberaxis = numberaxis - ((axistoplot-1)*float(4))
#        competition_bins = np.arange(0,10,1)
#        #twogroups = [0,group['stimulus competition'].median(),group['stimulus competition'].max()]
#        #competition_bins = np.arange(0.0,math.ceil(group['stimulus competition'].max()+0.5),group['stimulus competition'].max()/2.0)
#        temp_grouping = pd.cut(group['stimulus competition'], competition_bins)
#        # highcomp_select = group[group['stimulus competition']>max(group['stimulus competition'])/2.]['fixation duration']
#        # lowcomp_select = group[group['stimulus competition']<max(group['stimulus competition'])/2.]['fixation duration']
#        # size1 = len(highcomp_select)
#        # size2 = len(lowcomp_select )
#        # if size1>6 and size2>6:
#        #     highcomp_SF =  np.mean(highcomp_select)
#        #     lowcomp_SF = np.mean(lowcomp_select )
#        #     xvalueslist = [0,1]
#        #     axes[axistoplot-1,numberaxis].plot(xvalueslist,[lowcomp_SF,highcomp_SF])
#        if len(group['fixation duration'].dropna())>0:
#            #axes[axistoplot-1,numberaxis].plot(group['stimulus competition'], group['fixation duration'])
#            #group.plot(ax=axes[axistoplot-1,numberaxis], kind='scatter', x='stimulus competition', y='fixation duration',sharex=False)
#            group = group['fixation duration'].dropna()
#            #group = group.groupby(temp_grouping).filter(lambda x: len(x) > 5)
#            group.groupby(temp_grouping).mean().plot(ax=axes[axistoplot-1,numberaxis],sharex=False)
#            #yvalues = group.groupby(temp_grouping).mean()
#            #axes[axistoplot-1,numberaxis].plot(range(0,len(yvalues),1), yvalues)
#            axes[axistoplot-1,numberaxis].legend([i], loc=3)
#            axes[axistoplot-1,numberaxis].set_ylim([170,240])
#            axes[axistoplot-1,numberaxis].set_ylabel('SF duration')
#            axes[axistoplot-1,numberaxis].set_xlabel('Overlapping letters')
#            #axes[axistoplot-1,numberaxis].set_xticklabels([0,1])
#            axes[axistoplot-1,numberaxis].set_xticklabels(competition_bins)
#refix_after_wordskip = len(df_alldata_grouped_all[(df_alldata_grouped_all['wordskipped'] == True) & (df_alldata_grouped_all['refixated'] == True)])
#total_refix = len(df_alldata_grouped_all[df_alldata_grouped_all['refixated'] == True])
#print "% refix after wordskips:",refix_after_wordskip/float(total_refix)
