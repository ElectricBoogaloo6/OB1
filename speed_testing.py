__author__ = 'Sam van Leipsig'

import parameters as pm
import numpy as np

def bigram_activation(word,allBigrams,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    for bigram in allBigrams:
        # priming
        if bigram in lexicon_word_bigrams[word]:
            wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[bigram]
    return wordExcitationInput

def bigram_activation_set(word,allBigrams,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    for bigram in allBigrams:
        # priming
        if bigram in lexicon_word_bigrams[word]:
            wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[bigram]
    return wordExcitationInput

def bigram_activation_set_fast(word,allBigrams,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    intersect_list = allBigrams.intersection(lexicon_word_bigrams[word])
    for bigram in intersect_list:
        wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[bigram]
    return wordExcitationInput

def bigram_activation_set_fast2(word,allBigrams,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    intersect_list = list(allBigrams.intersection(lexicon_word_bigrams[word]))
    for bigram in intersect_list:
        wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[bigram]
    return wordExcitationInput

def monogram_activation_list(word,allMonograms,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    for monogram in allMonograms:
        if monogram in word:
            wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[monogram]
    return wordExcitationInput

# def monogram_activation_list2(word,allMonograms,lexicon_word_bigrams,lexicon,unitActivations):
#     return sum([(pm.bigram_to_word_excitation * unitActivations[monogram]) for monogram in lexicon[word] if monogram in allMonograms])
#     #return wordExcitationInput

def monogram_activation_set(word,allMonograms,lexicon_word_bigrams,unitActivations):
    wordExcitationInput = 0
    intersect_list = allMonograms.intersection(word)
    for monogram in intersect_list:
        wordExcitationInput+= pm.bigram_to_word_excitation * unitActivations[monogram]
    return wordExcitationInput


def word_activations(LEXICON_SIZE,lexicon_activewords_np,word_overlap_matrix,lexicon_normalized_word_inhibition,lexicon_word_activity_np,lexicon_word_inhibition_np):
    for word_ix in xrange(LEXICON_SIZE):
        inhibiting_words_np = np.where((lexicon_activewords_np == True) & (word_overlap_matrix[word_ix,:]>0))[0]
        norm_lexicon_word_activity = lexicon_normalized_word_inhibition * lexicon_word_activity_np[inhibiting_words_np]
        total_word_inhibition = np.dot(word_overlap_matrix[word_ix,inhibiting_words_np],norm_lexicon_word_activity)
        lexicon_word_inhibition_np[word_ix] = total_word_inhibition
    return lexicon_word_inhibition_np

def word_activations2(LEXICON_SIZE,lexicon_activewords_np,word_overlap_matrix,lexicon_normalized_word_inhibition,lexicon_word_activity_np,lexicon_word_inhibition_np):
    overlap_select = word_overlap_matrix[:,(lexicon_activewords_np == True)]
    lexicon_select = lexicon_word_activity_np[(lexicon_activewords_np == True)] * lexicon_normalized_word_inhibition
    lexicon_word_inhibition_np = np.dot(overlap_select,lexicon_select)
    # for word_ix in xrange(LEXICON_SIZE):
    #     total_word_inhibition = np.dot(overlap_select[word_ix,:],lexicon_select)
    #     lexicon_word_inhibition_np_2[word_ix] = total_word_inhibition
    # print "Equality",np.allclose(lexicon_word_inhibition_np,lexicon_word_inhibition_np_2)
    return lexicon_word_inhibition_np



