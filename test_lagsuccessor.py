import pandas as pd
import numpy as np


def get_lagsuccessor(freqbins, predbins):
    file_name = "Data/DF_exp_lag_successor.pkl"
    saccade_data = pd.io.pickle.read_pickle(file_name)


    df_SF_lagsucc_dict = {}
    df_SF_lagsucc_dict['lag length'] = saccade_data.groupby('previous length')['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ length'] = saccade_data.groupby('next length')['SFD'].agg(np.mean)
    SF_previousfreq_groups = pd.cut(saccade_data['previous freq'], freqbins)
    SF_nextfreq_groups = pd.cut(saccade_data['next freq'], freqbins)
    df_SF_lagsucc_dict['lag freq'] = saccade_data.groupby(SF_previousfreq_groups)['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ freq'] = saccade_data.groupby(SF_nextfreq_groups)['SFD'].agg(np.mean) #.drop('word frequency', 1)
    SF_previouspred_groups = pd.cut(saccade_data['previous pred'], predbins)
    SF_nextpred_groups = pd.cut(saccade_data['next pred'], predbins)
    df_SF_lagsucc_dict['lag pred'] = saccade_data.groupby(SF_previouspred_groups)['SFD'].agg(np.mean)
    df_SF_lagsucc_dict['succ pred'] = saccade_data.groupby(SF_nextpred_groups)['SFD'].agg(np.mean)
    return saccade_data


if __name__ == "__main__":
    freqbins = np.arange(0, 7, 1)  # not sure if correct
    predbins = np.arange(0.1, 1.2, 0.5)  # not sure if correct
    df_SF_lagsucc_dict = get_lagsuccessor(freqbins, predbins)
    print(df_SF_lagsucc_dict)