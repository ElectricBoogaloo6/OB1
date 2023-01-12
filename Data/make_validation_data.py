import pandas as pd
import pickle as pkl

validation_data = pd.read_hdf("Fixation_durations_complete.hdf")
dutch_data = pd.read_excel("L1ReadingData.xlsx")

with open("nederlands/word_freq.pkl") as f:
	freq_pred = pkl.load(f)

coldict = { "FFDr": "WORD_FIRST_FIXATION_DURATION",
            "SFD" : "WORD_SECOND_FIXATION_DURATION",
            "GZD" : "WORD_GAZE_DURATION",
            "TVT" : "WORD_TOTAL_READING_TIME",
            "nfp" : "WORD_FIRST_RUN_FIXATION_COUNT",
            "nsp" : "WORD_SECOND_RUN_FIXATION_COUNT",
            "word": "WORD"
           }

new = pd.DataFrame(columns=list(coldict.keys()))

for key in coldict.keys():
	new[key] = dutch_data[coldict[key]]

# new["nap"] = new["nfp"] + new["nsp"]

new.to_hdf("Fixation_durations_dutch.hdf","complete")
