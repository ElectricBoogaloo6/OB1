# CHANGED
__author__ = 'Sam van Leipsig'
import parameters as pm
import numpy as np
import math


## Basic
#---------------------------------------------------------------------------
def my_print(*args):
    if pm.print_all:
        for i in args:
            print(i)
        print("")


def is_similar_word_length(word1, word2):
    return abs(len(word1)-len(word2)) < (0.15 * max(len(word1), len(word2)))


# returns the word center position of a surrounding word
# word position is > 0 for following words, < 0 for previous words
def getMidwordPositionForSurroundingWord(word_position,rightWordEdgeLetterIndexes,leftWordEdgeLetterIndexes):
    wordCenterPosition=None
    if word_position>0:
        word_slice_length = rightWordEdgeLetterIndexes[word_position][1]-rightWordEdgeLetterIndexes[word_position][0]+1
        wordCenterPosition = rightWordEdgeLetterIndexes[word_position][0]+round(word_slice_length/2.0)-1
    elif word_position==-1:
        previous_word_length = leftWordEdgeLetterIndexes[-2][1]-leftWordEdgeLetterIndexes[-2][0]+1
        wordCenterPosition = leftWordEdgeLetterIndexes[-2][0]+round(previous_word_length/2.0)-1
    return wordCenterPosition


## Reading
#---------------------------------------------------------------------------

#should always ensure that the maximum possible value of the threshold doesn't exceed the maximum allowable word activity
def get_threshold(word,word_freq_dict,max_frequency,freq_p,max_threshold):  #word_pred_dict,pred_p
    # let threshold be fun of word freq. freq_p weighs how strongly freq is (1=max, then thresh. 0 for most freq. word; <1 means less havy weighting)
    word_threshold = max_threshold # from 0-1, inverse of frequency, scaled to 0(highest freq)-1(lowest freq)
    if pm.frequency_flag:
        try:
            word_frequency = word_freq_dict[word]
            word_threshold = max_threshold * ((max_frequency/freq_p) - word_frequency) / (max_frequency/freq_p)
        except KeyError:
            pass
        #GS Only lower threshold for short words
        if len(word) < 4:
            word_threshold = word_threshold/3
        #return (word_frequency_multiplier * word_predictability_multiplier) * (pm.start_nonlin - (pm.nonlin_scaler*(math.exp(pm.wordlen_nonlin*len(word)))))
        return (word_threshold/1.4)


def normalize_pred_values(pred_p,pred_values):
    max_predictability = 1.
    return ((pred_p * max_predictability) - pred_values) / (pred_p * max_predictability)


# Make sure saccError doesn't cause NextEyeposition > stimulus
def calc_saccade_error(saccade_distance,optimal_distance,saccErr_scaler,saccErr_sigma,saccErr_sigma_scaler):
    #TODO include fixdur, as in EZ and McConkie (smaller sacc error after longer fixations)
    saccade_error = (optimal_distance - abs(saccade_distance)) * saccErr_scaler
    saccade_error_sigma = saccErr_sigma + (abs(saccade_distance) * saccErr_sigma_scaler)
    saccade_error_norm = np.random.normal(saccade_error,saccade_error_sigma,1)
    if pm.use_saccade_error:
        return saccade_error_norm
    else:
        return 0.


def norm_distribution(mu,sigma,distribution_param,recognized):
    if recognized:
        return int(np.round(np.random.normal(mu-distribution_param,sigma,1)))
    else:
        return int(np.round(np.random.normal(mu,sigma,1)))
