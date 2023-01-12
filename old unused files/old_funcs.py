#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:15:33 2022

@author: nathanvaartjes
"""


### func in simulate experiments. For lowering threshold of length-based words.

lexicon_thresholds_np_backup=copy.deepcopy(lexicon_thresholds_np) #make backup for resetting later

stop_cycles = {}  # for monitoring when to remove lower thresholds. See line 671 (approx.)
lexicon_thresholds_np = copy.deepcopy(lexicon_thresholds_np_backup) #NV: reset thresholds array


if str(cur_cycle) in stop_cycles: #NV: if cycle is registered as stop cycle, revert thresholds (works for lists as well, so can handele multiple words on same stop_cycle)
    lexicon_thresholds_np[stop_cycles[str(cur_cycle)]] /= 0.8
    # TODO: what happens when 2 words are registered on the same stop cycle?

for word in words_above_threshold:
    # NV: if word is an affix, get all words that are similar length to the stem of the stimulus
    if word in affixes:
        stem_recognized=True
        break
        # word_lengths_to_be_matched.append(
        #    len(stimulus.replace('_', ''))-len(word.replace('_', '')))
        for ix, thresh in enumerate(lexicon_thresholds_np):
            if False: #is_similar_word_length(len(lexicon[ix].strip('_')), [len(stimulus.strip('_'))-len(word.strip('_'))]):
                #logging.debug(f'{word} induces reduced threshold in cycle {cur_cycle} for word {lexicon[ix]}')
                # remember that this index must be reversed in x cycles
                if ix in [item for sublist in list(stop_cycles.values()) for item in sublist]: #to flatten the list
                    continue #NV: if the word is already lowered, dont lower it more
                elif str(cur_cycle+3) in stop_cycles: #NV: if the cycle is already registered in the dict, append word
                    lexicon_thresholds_np[ix] *= 0.8
                    stop_cycles[str(cur_cycle+3)].append(ix)
                else: #NV: else, put ix on that cycle as new list.
                    lexicon_thresholds_np[ix] *= 0.8
                    stop_cycles[str(cur_cycle+3)] = [ix]
                    