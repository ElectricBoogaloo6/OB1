# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 13:11:05 2020

@author: ginas
"""

__author__ = 'SAM'
import time
import parameters as pm
import os

def create_name(filename):
    file_date = time.strftime(" %d_%m ")
    model_details = "SkewAttention SaccadeError%s Widthchange%s " %(pm.use_saccade_error,pm.use_attendposition_change)
    min_max_width = "AttWidth %s_%s" %(pm.min_attend_width,pm.max_attend_width)
    distribution_details  = "Mu%s_s%s_p%s " %(pm.mu,pm.sigma,pm.distribution_param)
    if pm.linear:
        threshold_details = "linear-%s Wlen%s %s Wfreq%s Wpred%s " %(pm.linear,pm.wordlen_lin,pm.start_lin,pm.wordfreq_p,pm.wordpred_p)
    else:
        threshold_details = "linear-%s Wlen%s_%s_%s Wfreq%s Wpred%s " %(pm.linear,pm.wordlen_nonlin,pm.start_nonlin,pm.nonlin_scaler,pm.wordfreq_p,pm.wordpred_p)
    activation_details = "b_inh%s w_inh%s w_exc%s decay%s" %(abs(pm.bigram_to_word_inhibition),pm.bigram_to_word_inhibition,pm.bigram_to_word_excitation,pm.decay)
    filepath2 = "C:/Users/ginas/Documents/Werk/OB1_SAM/Results"
    if not os.path.exists(filepath2):
        os.makedirs(filepath2)
    outputfile = str(filepath2 + filename + file_date + model_details + distribution_details + threshold_details + activation_details + ".dat")
    unrecognized = str(filepath2 + "unrecognized " + filename + file_date + model_details + distribution_details + threshold_details + ".dat")
    return outputfile, unrecognized
    

def create_name_josh(filename):

    filepath2 = "C:/Users/ginas/Documents/Werk/OB1_SAM/Results"

    outputfile = str(filepath2+".dat")
    unrecognized = str(filepath2+"_unrecognized"+".dat")
    return outputfile, unrecognized

#print create_name("short")