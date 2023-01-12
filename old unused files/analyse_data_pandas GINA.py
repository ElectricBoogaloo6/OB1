__author__ = 'Sam van Leipsig'

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
import pdb
import pandas as pd
import math
import os
import sys
from reading_common import get_stimulus_text_from_file
import read_saccade_data as exp
import analyse_data_plot as mod
import analyse_data_plot_qualitative as mod2
import analyse_data_transformation as trans
import parameters as pm



def get_results(input_text_filename,input_file_all_data,input_file_unrecognized_words):
    with open(input_file_all_data,"r") as f:
        with open(input_file_unrecognized_words,"r") as g:
            all_data = pickle.load(f)
            if not os.path.exists("plots/"):
                os.makedirs("plots/")

            ## Parameters
            freqbins  = np.arange(-0.0,8,2.0)
            predbins = np.arange(-0.0,1.01,0.333)
            distancebins = np.arange(-0.0,20,2.0)
            neighborbins = np.arange(0,10,3)


            if pm.use_boundary_task:
                df_boundary_task = pd.read_pickle('Data/boundary_task_DF.pkl')
                df_individual_words = df_boundary_task
                df_individual_words.rename(columns={'words':'foveal word','f':'freq'}, inplace=True)
            else:

#                if pm.language == "german":
#                    word_freq_dict, word_pred_values = get_freq_pred_files()
#                if pm.language == "dutch":
#                    word_freq_dict = pickle.load(open("Data/nederlands/freq.pkl"))
#                    word_pred_values = np.ones(len(textsplitbyspace))
#                    word_pred_values[:] = 0.1

                ## Get complete psc (add freq and pred)
                # German mode
                if pm.language == "german":
                    textfile = get_stimulus_text_from_file(input_text_filename)
                    textsplitbyspace = textfile.split(" ")
                # Dutch mode
                if pm.language == "dutch":
                    textsplitbyspace = pickle.load(open(input_text_filename))

                individual_words = []
                for word in textsplitbyspace:
                    if word.strip()!="":
                        individual_words.append(word.strip())
                df_individual_words = pd.DataFrame(individual_words)
                print(df_individual_words)

                if pm.language == "german":
                    df_freq_pred = exp.get_freq_and_pred()
                    if pm.use_grammar_prob:
                        df_freq_pred = exp.get_freq_and_syntax_pred()
                    if pm.uniform_pred:
                        df_freq_pred["pred"][:] = 0.25

                if pm.language == "dutch":
                    df_freq_pred = pickle.load(open("Data/nederlands/freq500_2.pkl","r"))  # TODO
                    df_freq_pred = pd.DataFrame.from_dict(df_freq_pred, orient="index", columns=["freq"])
                    df_freq_pred["pred"] = np.zeros(len(df_freq_pred))
                    df_freq_pred["pred"][:] = 0.1
                    df_freq_pred["word"] = df_freq_pred.index
                    df_freq_pred.index = range(0,len(df_freq_pred))
                #print(df_freq_pred)
                # TODO fix
                #import copy_reg
                df_freq_pred = df_freq_pred.iloc[0:len(df_individual_words),:]
                df_individual_words = pd.concat([df_individual_words,df_freq_pred],axis=1,join_axes=[df_individual_words.index])
                df_individual_words = df_individual_words.drop(['word'],1)
                df_individual_words.rename(columns={'0':'foveal word','f':'freq'}, inplace=True)
                df_individual_words_base = df_individual_words.copy()
                for i in range(0,pm.corpora_repeats):
                    df_individual_words = pd.concat([df_individual_words,df_individual_words_base],axis=0, ignore_index=True)

            # df_alldata.groupby('foveal word text index').filter(lambda x: len(x)>1 and sequential(x))

            ## Init dataframe
            df_alldata = pd.DataFrame(all_data)
            df_alldata['word length'] = df_alldata['foveal word'].map(len)
            df_alldata = trans.correct_wordskips(df_alldata)
            df_alldata = trans.correct_offset(df_alldata)
            df_alldata_no_regr = df_alldata[(df_alldata['regressed']==False)]  ## There are no refixations after a regression!

            ## Word measures by cycle, grouped by word length
            word_measures_bylen_dict = trans.make_word_measures_bylength(df_alldata)
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
            # print df_alldata[df_alldata['boundary task condition']>0][['refixated','refixation type','stimulus competition atshift','between word inhibition','foveal word', 'boundary task condition', 'actual recognized words','allocated word']]
            # pdb.set_trace()

            ## General fixation duration measures
            total_viewing_time = df_alldata.groupby(['foveal word text index'])[['fixation duration']].sum()
            gaze_durations = df_alldata_no_regr[['fixation duration','foveal word text index']].groupby(['foveal word text index']).sum()
            df_FD_only_regr = df_alldata[(df_alldata['regressed']==True)]['fixation duration']
            df_single_fixation, first_fixation, second_fixation = trans.make_number_fixations(df_alldata_no_regr)

            ## Qualitative effects
            df_alldata_grouped_neighbors = mod2.get_neighbor_data(df_alldata_no_regr)
            ## This threw an error so I used this fix to repair the file:
            ## https://stackoverflow.com/questions/556269/importerror-no-module-named-copy-reg-pickle/53942341#53942341
            mod2.plot_GD_byneighbors(df_alldata_grouped_neighbors,neighborbins)
            if pm.use_boundary_task:
                ## Boundary task
                df_SF_boundary_task, df_GD_boundary_task = mod2.preprocess_boundary_task(df_single_first_fixation, df_alldata_no_regr, df_individual_words)
                mod2.plot_boundary_task(df_SF_boundary_task,df_GD_boundary_task)


            ## Plot word activation measures per cycle
            # mod.plot_activity_percycle_bylenght(word_measures_bylen_dict['activity'], word_measures_bylen_dict['threshold'])
            # mod.plot_exc_inh_percycle_bylength(word_measures_bylen_dict['excitation'], word_measures_bylen_dict['inhibition'])
            # mod.plot_realactivity_decay_bylength(word_measures_bylen_dict['realactivation'], word_measures_bylen_dict['decay'])
            # ##Word activity
            #mod.plot_wordactivity_atshift_bylength(df_alldata_no_regr)
            #df_wordactivity = trans.make_df_wordactivity(all_data)
            #mod.plot_wordactivity_grouped(df_wordactivity,df_single_fixation,freqbins,predbins)
            #mod.word_activity_threshold(df_wordactivity)



            ## Unrecognized
            unrecognized_words = pickle.load(g)
            print(np.size(unrecognized_words), " = size of unrecognized words")
            print(np.shape(unrecognized_words), " = shape of unrecognized words")
            try:
                df_unrecognised_words = trans.unrecognized_words(unrecognized_words)
                mod.plot_unrecognizedwords(df_alldata,df_alldata_grouped_all,df_unrecognised_words)
                mod.plot_unrecognizedwords_bytype(df_alldata_grouped_all,df_unrecognised_words)
            except: #if there are no unrecognized words
                print("No unrecognized words")


            ## Saccade distance
            mod.plot_saccdistance(df_alldata_no_regr, exp.get_sacc_distance())
            mod.plot_saccdistance2(df_alldata_no_regr, exp.get_sacc_distance_bytype_simple())
            mod.plot_saccadedistance_bytype(df_alldata)
            mod.plot_saccerror(df_alldata)
            mod.plot_saccerror_type(df_alldata)

            ## landing position
            exp_landing_positions, exp_refixprob_by_rlpos, exp_SF_grpby_rlpos = exp.get_landing_postition()
            mod.plot_by_relative_landing_pos(df_single_fixation, df_alldata_grouped_all,exp_refixprob_by_rlpos,exp_SF_grpby_rlpos)
            print(df_alldata)
            print(exp_landing_positions)
            mod.plot_offset(df_alldata,exp_landing_positions)

            ## Other
            mod.plot_recognized_cycles(df_alldata)
            mod.plot_groupsize_distribution(df_alldata_grouped_all,freqbins,predbins)
            mod.plot_freqpred_bylength(df_alldata) #TODO put in groupsize distribution plot
            mod.plot_refix_types(df_alldata)
            mod.plot_attendwidth(df_alldata)
            mod.plot_word_similarity(df_alldata_grouped_all,13)
            mod.before_regression(df_alldata)
            # print df_alldata.describe()
            # print df_alldata[df_alldata['word predictability']>0.75]['word length'].mean()
            #mod.plot_overlapmatrix_by(df_alldata_grouped_all,freqbins)

            ## Sacctype prob
            exp_sacc_dict = exp.get_saccade_type_probabilities()
            mod.plot_saccadetype_probabilities(df_alldata_grouped_all,exp_sacc_dict)
            exp_sacctype_grpby_prob_dict = exp.get_grouped_sacctype_prob(freqbins,predbins)
            mod.plot_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins)
            ## FOR TESTING
            mod.sse_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins)


            ## Fixdur wordskips
            # mod.analyse_fixdur_aroundwordskip(df_alldata,df_unrecognised_words)
            # mod.fastvsslow_words(df_single_fixation)
            #print "test wordskipped",df_alldata[df_alldata['wordskipped'] == True]['wordskipped'], df_alldata[df_alldata['after wordskip']==True]['wordskip index']
            df_alldata_shortwords = df_alldata[df_alldata['word length']<6]
            #for name,wl_group in df_alldata_shortwords.groupby('word length'):

            # df_SF_wordskipeffect = df_single_fixation[['fixation duration','foveal word text index','word length','before wordskip','word frequency','after wordskip']]
            # df_SF_wordskipeffect = df_SF_wordskipeffect[df_SF_wordskipeffect['word length']<6]
            # for i in range(1,len(df_SF_wordskipeffect)-1):
            #     ix = df_SF_wordskipeffect.index.tolist()[i]
            #     ixplus1 = df_SF_wordskipeffect.index.tolist()[i+1]
            #     df_SF_wordskipeffect.loc[ix,'next frequency'] = df_SF_wordskipeffect.loc[ixplus1,'word frequency']
            #     df_SF_wordskipeffect.loc[ix,'next length'] = df_SF_wordskipeffect.loc[ixplus1,'word length']
            # df_SF_wordskipeffect = df_SF_wordskipeffect[1:]
            # print df_SF_wordskipeffect[df_SF_wordskipeffect['next frequency']>=4.0].groupby('before wordskip')['fixation duration'].mean()
            # print df_SF_wordskipeffect[df_SF_wordskipeffect['next frequency']<=2.0].groupby('before wordskip')['fixation duration'].mean()
            # print df_SF_wordskipeffect.groupby('after wordskip')['fixation duration'].mean()


            # ## Lag Successor
            wordlength_limit = 16
            ## TODO insert wordlength limit and gaze duration in get_lagsuccessor
            exp_SF_lagsucc_dict, exp_GD_lagsucc_dict = exp.get_lagsuccessor(freqbins,predbins)

            df_single_fixation = df_single_fixation.set_index('foveal word text index')
            mod.plot_lagsuccessor(df_alldata_no_regr,df_single_fixation,freqbins,predbins,exp_SF_lagsucc_dict, "SF", wordlength_limit)
            mod.plot_lagsuccessor(df_alldata_no_regr,gaze_durations,freqbins,predbins,exp_GD_lagsucc_dict, "GD", wordlength_limit)



            ## Fixation duration by group
            exp_FD_bylength_dict, exp_FD_byfreq_dict, exp_FD_bypred_dict = exp.get_FD_bygroup(freqbins,predbins)
            mod_FD_bylength_dict, mod_FD_byfreq_dict, mod_FD_bypred_dict  = trans.make_FD_bygroup(df_alldata_no_regr,df_alldata,df_single_fixation,freqbins,predbins)
            mod.plot_FD_bygroup(mod_FD_bylength_dict,mod_FD_byfreq_dict,mod_FD_bypred_dict,exp_FD_bylength_dict,exp_FD_byfreq_dict,exp_FD_bypred_dict)


            ## Fixation durations histograms
            exp_FD_dict = exp.get_saccade_durations()
            mod.plot_FD_hists(total_viewing_time,gaze_durations,df_single_fixation,first_fixation,second_fixation,df_FD_only_regr,exp_FD_dict)


def get_results_simulation(input_file_all_data,input_file_unrecognized_words):
    import parameters_exp as pm

    with open(input_file_all_data,"rb") as f:
        with open(input_file_unrecognized_words,"rb") as g:
            all_data = pickle.load(f, encoding="latin1") # For Python3
            #all_data = pickle.load(f)
            if not os.path.exists("plots/"):
                os.makedirs("plots/")

            ## Parameters
            freqbins  = np.arange(-0.0,8,2.0)
            predbins = np.arange(-0.0,1.01,0.333)
            distancebins = np.arange(-0.0,20,2.0)
            neighborbins = np.arange(0,10,3)

            # generate / read in stimuli list from file (fixed items for both experiments)
            if pm.use_sentence_task:
                stim = pd.read_table('./Stimuli/debug_Sentence_stimuli_all_csv.csv', sep=',')
                task = "Sentence"
            if pm.use_flanker_task:
                stim = pd.read_table('./Stimuli/debug_Flanker_stimuli_all_csv.csv', sep=',')
                task = "Flanker"

            #print(stim.head(10))
            stim['all'] = stim['all'].astype(str)
            individual_words = []
            lengtes=[]
            #textsplitbyspace = stim["all"].str.split(" ")
            textsplitbyspace = list(stim['all'].str.split(' ', expand=True).stack().unique())

            for word in textsplitbyspace:
                if word.strip() != "":
                    new_word = str(word.strip()) #For Python2
                    individual_words.append(new_word)
                    lengtes.append(len(word))

            print(individual_words)
            df_individual_words = pd.DataFrame(individual_words)
            print(df_individual_words)

            # load dictionaries (French Lexicon Project database) and generate list of individual words

            # df_freq_pred = exp.get_freq_and_pred_fr(task)
            # if pm.use_grammar_prob:
            #     print("not implemented yet")
            #     df_freq_pred = exp.get_freq_and_syntax_pred()
            # if pm.uniform_pred:
            #     df_freq_pred["pred"][:] = 0.25
            #
            #
            # df_freq_pred = df_freq_pred.iloc[0:len(df_individual_words),:]
            # df_individual_words = pd.concat([df_individual_words,df_freq_pred],axis=1,join_axes=[df_individual_words.index])
            # print(df_individual_words.head(10))
            # df_individual_words = df_individual_words.drop(['word'],1)
            # df_individual_words.rename(columns={'0':'foveal word','f':'freq'}, inplace=True)
            # df_individual_words_base = df_individual_words.copy()
            # for i in range(0,pm.corpora_repeats):
            #     df_individual_words = pd.concat([df_individual_words,df_individual_words_base],axis=0, ignore_index=True)

            ## Init dataframe
            df_alldata = pd.DataFrame(all_data)
            print(df_alldata)
            print(task)
            df_alldata.to_pickle('alldata_' + task + ".pkl")
            df_alldata['word length'] = df_alldata['target'].map(len)
            df_alldata = trans.correct_wordskips(df_alldata)
            df_alldata = trans.correct_offset(df_alldata)
            df_alldata_no_regr = df_alldata[(df_alldata['regressed']==False)]  ## There are no refixations after a regression!

            ## Word measures by cycle, grouped by word length
            word_measures_bylen_dict = trans.make_word_measures_bylength(df_alldata)
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

            ## General fixation duration measures
            total_viewing_time = df_alldata.groupby(['foveal word text index'])[['fixation duration']].sum()
            gaze_durations = df_alldata_no_regr[['fixation duration','foveal word text index']].groupby(['foveal word text index']).sum()
            df_FD_only_regr = df_alldata[(df_alldata['regressed']==True)]['fixation duration']
            df_single_fixation, first_fixation, second_fixation = trans.make_number_fixations(df_alldata_no_regr)

            ## Qualitative effects
            df_alldata_grouped_neighbors = mod2.get_neighbor_data(df_alldata_no_regr)
            ## This threw an error so I used this fix to repair the file:
            ## https://stackoverflow.com/questions/556269/importerror-no-module-named-copy-reg-pickle/53942341#53942341
            mod2.plot_GD_byneighbors(df_alldata_grouped_neighbors,neighborbins)
            if pm.use_boundary_task:
                ## Boundary task
                df_SF_boundary_task, df_GD_boundary_task = mod2.preprocess_boundary_task(df_single_first_fixation, df_alldata_no_regr, df_individual_words)
                mod2.plot_boundary_task(df_SF_boundary_task,df_GD_boundary_task)


            # Plot word activation measures per cycle
            mod.plot_activity_percycle_bylenght(word_measures_bylen_dict['activity'], word_measures_bylen_dict['threshold'])
            mod.plot_exc_inh_percycle_bylength(word_measures_bylen_dict['excitation'], word_measures_bylen_dict['inhibition'])
            mod.plot_realactivity_decay_bylength(word_measures_bylen_dict['realactivation'], word_measures_bylen_dict['decay'])
            ##Word activity
            mod.plot_wordactivity_atshift_bylength(df_alldata_no_regr)
            df_wordactivity = trans.make_df_wordactivity(all_data)
            mod.plot_wordactivity_grouped(df_wordactivity,df_single_fixation,freqbins,predbins)
            mod.word_activity_threshold(df_wordactivity)

            ## Unrecognized
            unrecognized_words = pickle.load(g)
            print(np.size(unrecognized_words), " = size of unrecognized words")
            print(np.shape(unrecognized_words), " = shape of unrecognized words")
            try:
                df_unrecognised_words = trans.unrecognized_words(unrecognized_words)
                mod.plot_unrecognizedwords(df_alldata,df_alldata_grouped_all,df_unrecognised_words)
                mod.plot_unrecognizedwords_bytype(df_alldata_grouped_all,df_unrecognised_words)
            except: #if there are no unrecognized words
                print("No unrecognized words")


            ## Saccade distance
            mod.plot_saccdistance(df_alldata_no_regr, exp.get_sacc_distance())
            mod.plot_saccdistance2(df_alldata_no_regr, exp.get_sacc_distance_bytype_simple())
            mod.plot_saccadedistance_bytype(df_alldata)
            mod.plot_saccerror(df_alldata)
            mod.plot_saccerror_type(df_alldata)

            ## landing position
            exp_landing_positions, exp_refixprob_by_rlpos, exp_SF_grpby_rlpos = exp.get_landing_postition()
            mod.plot_by_relative_landing_pos(df_single_fixation, df_alldata_grouped_all,exp_refixprob_by_rlpos,exp_SF_grpby_rlpos)
            print(df_alldata)
            print(exp_landing_positions)
            mod.plot_offset(df_alldata,exp_landing_positions)

            ## Other
            mod.plot_recognized_cycles(df_alldata)
            mod.plot_groupsize_distribution(df_alldata_grouped_all,freqbins,predbins)
            mod.plot_freqpred_bylength(df_alldata) #TODO put in groupsize distribution plot
            mod.plot_refix_types(df_alldata)
            mod.plot_attendwidth(df_alldata)
            mod.plot_word_similarity(df_alldata_grouped_all,13)
            mod.before_regression(df_alldata)
            # print df_alldata.describe()
            # print df_alldata[df_alldata['word predictability']>0.75]['word length'].mean()
            #mod.plot_overlapmatrix_by(df_alldata_grouped_all,freqbins)

            ## Sacctype prob
            exp_sacc_dict = exp.get_saccade_type_probabilities()
            mod.plot_saccadetype_probabilities(df_alldata_grouped_all,exp_sacc_dict)
            exp_sacctype_grpby_prob_dict = exp.get_grouped_sacctype_prob(freqbins,predbins)
            mod.plot_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins)
            ## FOR TESTING
            mod.sse_sacctypeprob_bygroup(df_alldata_grouped_all,exp_sacctype_grpby_prob_dict,freqbins,predbins)


            ## Fixdur wordskips
            # mod.analyse_fixdur_aroundwordskip(df_alldata,df_unrecognised_words)
            # mod.fastvsslow_words(df_single_fixation)
            #print "test wordskipped",df_alldata[df_alldata['wordskipped'] == True]['wordskipped'], df_alldata[df_alldata['after wordskip']==True]['wordskip index']
            df_alldata_shortwords = df_alldata[df_alldata['word length']<6]
            #for name,wl_group in df_alldata_shortwords.groupby('word length'):

            # df_SF_wordskipeffect = df_single_fixation[['fixation duration','foveal word text index','word length','before wordskip','word frequency','after wordskip']]
            # df_SF_wordskipeffect = df_SF_wordskipeffect[df_SF_wordskipeffect['word length']<6]
            # for i in range(1,len(df_SF_wordskipeffect)-1):
            #     ix = df_SF_wordskipeffect.index.tolist()[i]
            #     ixplus1 = df_SF_wordskipeffect.index.tolist()[i+1]
            #     df_SF_wordskipeffect.loc[ix,'next frequency'] = df_SF_wordskipeffect.loc[ixplus1,'word frequency']
            #     df_SF_wordskipeffect.loc[ix,'next length'] = df_SF_wordskipeffect.loc[ixplus1,'word length']
            # df_SF_wordskipeffect = df_SF_wordskipeffect[1:]
            # print df_SF_wordskipeffect[df_SF_wordskipeffect['next frequency']>=4.0].groupby('before wordskip')['fixation duration'].mean()
            # print df_SF_wordskipeffect[df_SF_wordskipeffect['next frequency']<=2.0].groupby('before wordskip')['fixation duration'].mean()
            # print df_SF_wordskipeffect.groupby('after wordskip')['fixation duration'].mean()


            # ## Lag Successor
            wordlength_limit = 16
            ## TODO insert wordlength limit and gaze duration in get_lagsuccessor
            exp_SF_lagsucc_dict, exp_GD_lagsucc_dict = exp.get_lagsuccessor(freqbins,predbins)

            df_single_fixation = df_single_fixation.set_index('foveal word text index')
            mod.plot_lagsuccessor(df_alldata_no_regr,df_single_fixation,freqbins,predbins,exp_SF_lagsucc_dict, "SF", wordlength_limit)
            mod.plot_lagsuccessor(df_alldata_no_regr,gaze_durations,freqbins,predbins,exp_GD_lagsucc_dict, "GD", wordlength_limit)



            ## Fixation duration by group
            exp_FD_bylength_dict, exp_FD_byfreq_dict, exp_FD_bypred_dict = exp.get_FD_bygroup(freqbins,predbins)
            mod_FD_bylength_dict, mod_FD_byfreq_dict, mod_FD_bypred_dict  = trans.make_FD_bygroup(df_alldata_no_regr,df_alldata,df_single_fixation,freqbins,predbins)
            mod.plot_FD_bygroup(mod_FD_bylength_dict,mod_FD_byfreq_dict,mod_FD_bypred_dict,exp_FD_bylength_dict,exp_FD_byfreq_dict,exp_FD_bypred_dict)


            ## Fixation durations histograms
            exp_FD_dict = exp.get_saccade_durations()
            mod.plot_FD_hists(total_viewing_time,gaze_durations,df_single_fixation,first_fixation,second_fixation,df_FD_only_regr,exp_FD_dict)
