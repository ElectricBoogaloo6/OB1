# -*- coding: utf-8 -*-
__author__ = 'Phillip Kersten, adapted from Sam van Leipsig'

from time import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
import pandas as pd
import analyse_data_plot as mod
import analyse_data_plot_qualitative as mod2
from reading_common import get_stimulus_text_from_file
import read_saccade_data as exp
import analyse_data_transformation as trans
from parameters import return_params
import matplotlib.lines as mlines

pm=return_params()

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_scores(input_text_filename,all_data,unrecognized_words):
#    with open(input_file_all_data,"r") as f:
#        all_data = pickle.load(f)
#    with open(input_file_unrecognized_words,"r") as g:
#        unrecognized_words = pickle.load(g)
    ## Parameters
    freqbins  = np.arange(-0.0,8,2.0)
    predbins = np.arange(-0.0,1.01,0.333)
    distancebins = np.arange(-0.0,20,2.0)
    neighborbins = np.arange(0,10,3)


    ## Get complete psc (add freq and pred)
    textfile = get_stimulus_text_from_file("PSC/" + input_text_filename + '.txt')
    individual_words = []
    textsplitbyspace = textfile.split(" ")
    for word in textsplitbyspace:
        if word.strip()!="":
            individual_words.append(word.strip())
    df_individual_words = pd.DataFrame(individual_words)

    if pm.use_grammar_prob:
        df_freq_pred = exp.get_freq_and_syntax_pred()
    else:
        df_freq_pred = exp.get_freq_and_pred()

    if pm.uniform_pred:
        df_freq_pred["pred"][:] = 0.25

    df_freq_pred = df_freq_pred.iloc[0:len(df_individual_words),:]
    df_individual_words = pd.concat([df_individual_words,df_freq_pred],axis=1,join_axes=[df_individual_words.index])
    df_individual_words = df_individual_words.drop(['word'],1)
    df_individual_words.rename(columns={'0':'foveal word','f':'freq'}, inplace=True)
    df_individual_words_base = df_individual_words.copy()
    for i in range(0,pm.corpora_repeats):
        df_individual_words = pd.concat([df_individual_words,df_individual_words_base],axis=0, ignore_index=True)

    ## Init dataframe
    df_alldata = pd.DataFrame(all_data)
    df_alldata['word length'] = df_alldata['foveal word'].map(len)
    df_alldata = trans.correct_wordskips(df_alldata)
    df_alldata = trans.correct_offset(df_alldata)
    df_alldata_no_regr = df_alldata[(df_alldata['regressed']==False)]  ## There are no refixations after a regression!

    ## Word measures by cycle, grouped by word length
    # not necessary??  # word_measures_bylen_dict = trans.make_word_measures_bylength(df_alldata)
    df_alldata = df_alldata.drop(['fixation word activities np'],1)
    df_alldata_no_regr['foveal word text index2'] = df_alldata_no_regr['foveal word text index']
    df_SF = df_alldata_no_regr.groupby(['foveal word text index']).filter(lambda x: len(x)==1)

    ## Select first fixation and single fixations, use sequential to select first pass only
    df_fixations_sequential = df_alldata_no_regr.groupby(['foveal word text index']).filter(lambda x: trans.sequential(x))
    singlefirst_fixation_grouped =  df_fixations_sequential.groupby(['foveal word text index'])
    singlefirst_fixation_selection =  singlefirst_fixation_grouped.apply(lambda x: x.index[0]).values
    df_single_first_fixation = df_alldata_no_regr.loc[singlefirst_fixation_selection,:]


    ## Create complete dataset including wordskips
    df_individual_words.reset_index ## index is now the same as foveal word text index
    df_alldata_to_group = df_alldata.drop(['word length','foveal word','recognized words indices','fixation word activities', 'word activities per cycle', 'stimulus'],1)
    df_alldata_grouped_max = df_alldata_to_group.groupby('foveal word text index', as_index= True).max()
    df_alldata_grouped_all = pd.concat([df_individual_words,df_alldata_grouped_max], axis=1, join_axes=[df_individual_words.index])
    df_alldata_grouped_all['wordskipped'].fillna(True, inplace=True)
    replaceNA = {'regressed': False, 'refixated': False,'forward':False,'after wordskip':False,'before wordskip':False}
    df_alldata_grouped_all.fillna(replaceNA,inplace=True)
    df_alldata_grouped_all.rename(columns={0:'foveal word'}, inplace=True)
    df_alldata_grouped_all['word length'] = df_alldata_grouped_all['foveal word'].map(len)

    # print df_alldata.columns.values

    ## General fixation duration measures
    total_viewing_time = df_alldata.groupby(['foveal word text index'])[['fixation duration']].sum()
    gaze_durations = df_alldata_no_regr[['fixation duration','foveal word text index']].groupby(['foveal word text index']).sum()
    df_FD_only_regr = df_alldata[(df_alldata['regressed']==True)]['fixation duration']
    df_single_fixation, first_fixation, second_fixation = trans.make_number_fixations(df_alldata_no_regr)

    df_single_fixation = df_single_fixation.set_index('foveal word text index')

    exp_FD_dict = exp.get_saccade_durations()

    ## Get distance between curves in plot
    sse = 0
    total_divergence = 0
    sses = {}
    divergences = {}
    simulation = [total_viewing_time["fixation duration"],
                  gaze_durations['fixation duration'],
                  df_single_fixation['fixation duration'],
                  first_fixation,
                  second_fixation,
                  df_FD_only_regr
                  ]
    experiment = [exp_FD_dict["TVT"],
                  exp_FD_dict['GZD'],
                  exp_FD_dict['SFD'],
                  exp_FD_dict['FFD'],
                  exp_FD_dict['SecondFD'],
                  exp_FD_dict['RD']
                  ]
    names = ["total viewing time",
             "Gaze duration",
             "Single fixations",
             "First fixation duration",
             "Second fixation duration",
             "Regression"
             ]

    with open("experiment.pkl","w") as f:
        pickle.dump(experiment,f)

    with open("simulation.pkl","w") as f:
        pickle.dump(simulation,f)

    t = time()
    if pm.include_sacc_type_sse:
        exp_sacctype_grpby_prob_dict = exp.get_grouped_sacctype_prob(freqbins,predbins)
        sse_pred = mod.sse_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins)
        with open("sse_pred.txt", "a") as f:
            f.write(str(t)+str(int(sse_pred))+"\n")

    if pm.include_sacc_dist_sse:
        sse_dist = mod.sse_saccdistance(df_alldata_no_regr, exp.get_sacc_distance())
        with open("sse_saccdist.txt", "a") as f:
            f.write(str(t)+str(int(sse_dist))+"\n")

    plt.close()
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')
    ax = ax.ravel()
    legends = []
    i = 0
    for sim_, exp_, name in zip(simulation,experiment, names):

        min_ = min([exp_.min(), sim_.min()])
        max_ = max([exp_.max(), sim_.max()])
        if pm.discretization == "bin":
            resolution_ms = 25  # specify resolution of 25 ms
            n_bins = max_/resolution_ms
            bins = np.linspace(min_, max_, n_bins)  # bins for SSE
            x_index = np.digitize(exp_, bins)
            y_index = np.digitize(sim_, bins)
            x = np.bincount(x_index)
            y = np.bincount(y_index, minlength=len(x))
        if pm.discretization == "kde":
            X = np.mgrid[min_:max_+1:25]  # discretize data in 25ms steps (for kde)
            positions = X.ravel()
            values_x = sim_
            values_y = exp_
            print("-----------------values x------------------")
            print(values_x)
            print("-----------------values y------------------")
            print(values_y)
            # Workaround to make parametertuning possible in case there is no data because e.g. no regressions have been made
            try:
                kernel_x = stats.gaussian_kde(values_x)
            except:
                print("Error: empty dataframe for "+name+"replacing x with inverse y")
                kernel_x = stats.gaussian_kde(-values_y)
                kernel_y = stats.gaussian_kde(values_y)
            # Set bandwidth like in the original plotting method
            band_width = 0.31
            kernel_x.set_bandwidth(band_width)
            kernel_y.set_bandwidth(band_width)
            x = np.reshape(kernel_x(positions).T, X.shape)
            y = np.reshape(kernel_y(positions).T, X.shape)
        sses[name] = sum(map(lambda x_: (x_[0]-x_[1])**2, zip(x, y)))
        sse += sses[name]
        # KL divergence only possible with kernel density estimation
        if pm.discretization == "kde":
            divergences[name] = kl_divergence(x,y)
            total_divergence += divergences[name]
        plot = True
        if plot:
            ax[i].plot(x, "b")  # Experiment
            ax[i].plot(y, "r")  # Simulation
            ax[i].set_title(name+": \n"+str(round(sses[name], 3)))
            i += 1
    blue_line = mlines.Line2D([], [], color="blue")
    red_line = mlines.Line2D([], [], color="red")
    plt.figlegend(handles=[red_line, blue_line], labels=["Simulation", "Experiment"], loc='upper right')

    # Add specific sse's if wanted
    if pm.include_sacc_type_sse:
        sse += sse_pred
    if pm.include_sacc_dist_sse:
        sse += sse_dist

    suptitle = plt.suptitle("SSE: "+str(round(sse, 4)), y=1.02)
    fig.tight_layout()
    plt.savefig("test_density"+str(int(t))+".png", bbox_extra_artists=(suptitle, ), bbox_inches="tight", dpi=300)
    if pm.tuning_measure == "KL":
        if not any(pm.objective):
            return total_divergence
        else:
            return divergences[pm.objective]
    if pm.tuning_measure == "SSE":
        if not any(pm.objective):
            return sse
        else:
            return sses[pm.objective]
