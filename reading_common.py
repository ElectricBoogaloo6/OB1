from __future__ import division

import math
import codecs
import re
import itertools

from parameters import return_params

pm = return_params()


def getStimulusSpacePositions(stimulus):  # NV: get index of spaces in stimulus
    stimulus_space_positions = []
    stimulus_space_positions_append = stimulus_space_positions.append
    for letter_position in range(len(stimulus)):
        if stimulus[letter_position] == " ":
            stimulus_space_positions_append(letter_position)

    return stimulus_space_positions


def getNgramEdgePositionWeight(ngram, ngramLocations, stimulus_space_locations):
    ngramEdgePositionWeight = 0.5  # This is just a default weight; in many cases, it's changed
    # to 1 or 2, as can be seen below.
    max_weight = 2.
    # print stimulus_space_locations
    if len(ngram) == 2:
        first = ngramLocations[0]
        second = ngramLocations[1]

        if (first-1) in stimulus_space_locations and (second+1) in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        elif (first+1) in stimulus_space_locations and (second-1) in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        elif (first-1) in stimulus_space_locations or (second+1) in stimulus_space_locations:
            ngramEdgePositionWeight = 1.
        elif (first+1) in stimulus_space_locations or (second-1) in stimulus_space_locations:
            ngramEdgePositionWeight = 1.
    else:
        letter_location = ngramLocations
        # One letter word
        if letter_location-1 in stimulus_space_locations and letter_location+1 in stimulus_space_locations:
            ngramEdgePositionWeight = max_weight
        # letter at the edge
        elif letter_location-1 in stimulus_space_locations or letter_location+1 in stimulus_space_locations:
            ngramEdgePositionWeight = 1.

    return ngramEdgePositionWeight


def stringToBigramsAndLocations(stimulus, is_prefix, is_suffix):
    stimulus_space_positions = getStimulusSpacePositions(stimulus)
    stimulus = "_"+stimulus+"_"  # GS try to recognize small words better, does not work doing this
    # For the current stimulus, bigrams will be made. Bigrams are only made
    # for letters that are within a range of 4 from each other; (gap=3)

    # Bigrams that contain word boundary letters have more weight.
    # This is done by means of locating spaces in stimulus, and marking
    # letters around space locations (as well as spaces themselves), as
    # indicators of more bigram weight.

    """Returns list with all unique open bigrams that can be made of the stim, and their respective locations 
    (called 'word' for historic reasons), restricted by the maximum gap between two intervening letters."""
    allBigrams = []
    allBigrams_append = allBigrams.append
    bigramsToLocations = {}
    gap = pm.bigram_gap  # None = no limit
    if gap == None:
        for first in range(len(stimulus) - 1):
            if(stimulus[first] == " "):
                continue
            for second in range(first + 1, len(stimulus)):
                if(stimulus[second] == " "):
                    break
                bigram = stimulus[first]+stimulus[second]
                if bigram != '  ':
                    if not bigram in allBigrams:
                        allBigrams_append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(
                        bigram, (first, second), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second, bigramEdgePositionWeight)]
    else:
        for first in range(len(stimulus) - 1):

            # NV: this code implant is meant to insert special suffix bigrams in bigram list
            #TODO: does not work for prefixes yet
            if(stimulus[first] == " "):  # NV: means that first letter is index 1 or last+1
                if first == 1:  # NV: if it is in the beginning of word
                    if is_suffix:
                        continue #dont do the special _ at beginning of word if is affix
                    second_alt = 2  # NV: _alt not to interfere with loop variables
                    bigram = '_'+stimulus[second_alt]
                    if not bigram in allBigrams:
                        allBigrams_append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(
                        bigram, (first, second_alt), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second_alt, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second_alt, bigramEdgePositionWeight)]
                    continue
                elif first == len(stimulus)-2:  # NV: if first letter is the end space
                    first_alt = -3  # NV: index of last letter
                    # NV: get the actual index (you do +, because first alt is a negative number)
                    first_alt = len(stimulus)+first_alt
                    second_alt = -2  # NV: index of space after last letter
                    second_alt = len(stimulus)+second_alt
                    bigram = stimulus[first_alt]+'_'
                    if not bigram in allBigrams:
                        allBigrams_append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(
                        bigram, (first_alt, second_alt), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first_alt, second_alt, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first_alt, second_alt, bigramEdgePositionWeight)]
                    continue

            # NV:pick letter between first+1 and first+1+gap+1 (end of bigram max length), as long as that is smaller than end of word
            for second in range(first + 1, min(first+1+gap+1, len(stimulus))):
                if(stimulus[second] == " "):  # NV: if that is second lettter, you know you have reached the end of possible bigrams
                    # NV: break out of second loop if second stim is __. This means symbols before word, or when end of word is reached.
                    break
                bigram = stimulus[first]+stimulus[second]
                if bigram != '  ':
                    if not bigram in allBigrams:
                        allBigrams_append(bigram)
                    bigramEdgePositionWeight = getNgramEdgePositionWeight(
                        bigram, (first, second), stimulus_space_positions)
                    if(bigram in bigramsToLocations.keys()):
                        bigramsToLocations[bigram].append((first, second, bigramEdgePositionWeight))
                    else:
                        bigramsToLocations[bigram] = [(first, second, bigramEdgePositionWeight)]

# GS: test of dit werkt, dan geen monograms, die worden later toch weggefilterd
#    """Also add monograms"""
#    for position in range(len(stimulus)):
#        monogram=stimulus[position]
#        if(monogram==" "):
#            continue
#
#        if not monogram in allBigrams:
#            allBigrams_append(monogram)
#
#        monogramEdgePositionWeight = getNgramEdgePositionWeight(monogram, position,stimulus_space_positions)
#        if monogram in bigramsToLocations.keys():
#            bigramsToLocations[monogram].append((position,monogramEdgePositionWeight))
#        else:
#            bigramsToLocations[monogram]=[(position,monogramEdgePositionWeight)]
    # edge_bigrams = ["_"+stimulus[2], stimulus[-3]+"_"]  # NV: added first and last bigram for affix recognition.
    return (allBigrams, bigramsToLocations)


def get_attention_skewed(attentionWidth, attention_eccentricity, attention_skew):
    # Remember to remove the abs with calc functions
    if attention_eccentricity < 0:
        # Attention left
        attention = 1.0/(attentionWidth)*math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                  (2*pow(attentionWidth/attention_skew, 2))) + 0.25
    else:
        # Attention right
        attention = 1.0/(attentionWidth)*math.exp(-(pow(abs(attention_eccentricity), 2)) /
                                                  (2*pow(attentionWidth, 2))) + 0.25
    return attention


def calcAcuity(eye_eccentricity, letPerDeg):
    # Parameters from Harvey & Dumoulin (2007); 35.55556 is to make acuity at 0 degs eq. to 1
    return (1/35.555556)/(0.018*(eye_eccentricity*letPerDeg+1/0.64))


# parameters shift and amount of cycles are used only for reading
def calcBigramExtInput(bigram, bigrLocsStimulus, EyePosition, AttentionPosition, attendWidth, shift=True, amount_of_cycles=0):
    sumExtInput = 0
    # Here we look up all instances of same bigram. Act of all is summed (this is somewhat of a questionable assumption, perhaps max() would be better
    locations = bigrLocsStimulus[bigram]
    # todo check if locations weights multiplier is correct fixated words
    for bigram_letter_locations in locations:
        bigram_locations_weight_multiplier = bigram_letter_locations[2]

        # Bigram activity depends on distance of bigram letters to the centre of attention and fixation
        # and left/right is skewed using negative/positve att_ecc

        attention_eccentricity1 = bigram_letter_locations[0]-AttentionPosition
        attention_eccentricity2 = bigram_letter_locations[1]-AttentionPosition

        eye_eccentricity1 = abs(bigram_letter_locations[0]-EyePosition)
        eye_eccentricity2 = abs(bigram_letter_locations[1]-EyePosition)

        attention1 = get_attention_skewed(attendWidth, attention_eccentricity1, pm.attention_skew)
        attention2 = get_attention_skewed(attendWidth, attention_eccentricity2, pm.attention_skew)

        # Parameters from Harvey & Dumoulin (2007); 35.55556 is to make acuity at 0 degs eq. to 1
        visualAccuity1 = calcAcuity(eye_eccentricity1, pm.letPerDeg)
        visualAccuity2 = calcAcuity(eye_eccentricity2, pm.letPerDeg)

        extInput1 = attention1*visualAccuity1
        extInput2 = attention2*visualAccuity2
        extInput = math.sqrt(extInput1*extInput2)

        sumExtInput = sumExtInput+extInput * bigram_locations_weight_multiplier

    return sumExtInput


def calcMonogramExtInput(monogram, bigrLocsStimulus, EyePosition, AttentionPosition, attendWidth, shift=True, amount_of_cycles=0):
    sumExtInput = 0
    # Here we look up all instances of same monogram. Act of all is summed
    locations = bigrLocsStimulus[monogram]

    for monogram_position in locations:
        monogram_locations_weight_multiplier = monogram_position[1]
        # Monogram activity depends on distance of bigram letters to the centre of attention and fixation

        attention_eccentricity1 = monogram_position[0]-AttentionPosition
        eye_eccentricity1 = abs(monogram_position[0]-EyePosition)

        attention1 = get_attention_skewed(attendWidth, attention_eccentricity1, pm.attention_skew)
        visualAccuity1 = calcAcuity(eye_eccentricity1, pm.letPerDeg)

        extInput = attention1*visualAccuity1
        sumExtInput += extInput * monogram_locations_weight_multiplier

    return sumExtInput


def calcContrast(monogram_postition, EyePosition, AttentionPosition, attendWidth):
    attention_eccentricity1 = monogram_postition-AttentionPosition
    eye_eccentricity1 = abs(monogram_postition-EyePosition)
    attention1 = get_attention_skewed(attendWidth, attention_eccentricity1, pm.attention_skew)
    visualAccuity1 = calcAcuity(eye_eccentricity1, pm.letPerDeg)

    return attention1*visualAccuity1


# parameters shift and amount of cycles are used only for reading
# this is only used to calculate where to move next when forward saccade
def calcMonogramAttentionSum(positionStart, numberOfLetters, EyePosition, AttentionPosition, attendWidth, foveal_word, shift=True):
    sumAttentionLetters = 0
    for letter_location in range(positionStart, (positionStart+numberOfLetters)+1):
        monogram_locations_weight_multiplier = 0.5
        if foveal_word:
            if letter_location == positionStart+numberOfLetters:
                monogram_locations_weight_multiplier = 2.
        elif letter_location == positionStart or letter_location == positionStart+numberOfLetters:
            monogram_locations_weight_multiplier = 2.

        # Monogram activity depends on distance of monogram letters to the centre of attention and fixation
        attention_eccentricity1 = letter_location - AttentionPosition
        eye_eccentricity1 = abs(letter_location - EyePosition)

        attention1 = get_attention_skewed(attendWidth, attention_eccentricity1, pm.attention_skew)
        visualAccuity1 = calcAcuity(eye_eccentricity1, pm.letPerDeg)

        sumAttentionLetters += (attention1 * visualAccuity1) * monogram_locations_weight_multiplier

    return sumAttentionLetters


def calc_word_attention_right(rightWordEdgeLetterIndexes, EyePosition, AttentionPosition, attendWidth, recognized_position_flag, fixation, salience_position):
    # MM: calculate list of attention wgts for all words in stimulus to right of fix.
    word_attention_right = []
    word_attention_right_append = word_attention_right.append
    AttentionPosition += round(salience_position*attendWidth)
    for i, wordEdges in enumerate(rightWordEdgeLetterIndexes):
        crtWordMonogramAttentionSum = None
        word_start_edge = wordEdges[0]
        word_end_edge = wordEdges[1]

        # Foveal word flag necessary for lateral inhibition calculation
        if len(rightWordEdgeLetterIndexes) >= 2:
            foveal_word = True if i == 0 else False
        else:
            foveal_word = False

        # if at last letter (right edge) of fixated word (represented by -1,-1 in word_..._edge)
        # set attention wghts for (nonexisting) right part of fixated word to 0
        # TODO can be removed, just have start - end = 0
        if word_start_edge == -1 and word_end_edge == -1:
            crtWordMonogramAttentionSum = 0
        else:
            crtWordMonogramAttentionSum = calcMonogramAttentionSum(
                word_start_edge, word_end_edge - (word_start_edge), EyePosition, AttentionPosition, attendWidth, foveal_word)
        word_attention_right_append(crtWordMonogramAttentionSum)
    return word_attention_right


def select_mono_bigrams(allNgrams):
    allMonograms = []
    allBigrams = []
    for ngram in allNgrams:
        if(len(ngram) == 2):
            allBigrams.append(ngram)
        else:
            allMonograms.append(ngram)
    return allMonograms, allBigrams


def get_stimulus_text_from_file(filepath):
    # Need to determine encoding of original text
    with codecs.open(filepath, encoding='ISO-8859-1', errors='strict') as text_to_read:
        textfile = text_to_read.readline().lower()
        #textfile  = textfile.encode("utf-8", errors="strict")
        #textfile_clean = textfile.decode("ISO-8859-1",errors = "strict")
        #textfile = unidecode(textfile)

        # remove non-alphabet_numeric symbols
        my_re = re.compile('([^\s\w]|_)+', re.UNICODE)
        textfile_clean = my_re.sub('', textfile)
        # print textfile_clean
    return textfile_clean


def get_stimulus_text_from_file2(filepath):
    # Need to determine encoding of original text
    with codecs.open(filepath, encoding='ISO-8859-1', errors='strict') as text_to_read:
        textfile = text_to_read.readline().lower()
    return textfile
