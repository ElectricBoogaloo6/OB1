# -*- coding: UTF-8 -*-

# The purpose of this function is to load the parameters of interest with appropriate bounds to change during tuning
def get_params(pm):
	parameters = []
	bounds = []
	names = []

	parameters.append(pm.decay)
	bounds.append((-0.95,-0.01))
	names.append("decay")

	parameters.append(pm.bigram_to_word_excitation)
	bounds.append((0, None))
	names.append("bigram_to_word_excitation")

	parameters.append(pm.bigram_to_word_inhibition)
	bounds.append((None, 0))
	names.append("bigram_to_word_inhibition")

	parameters.append(pm.word_inhibition)
	bounds.append((None, 0))
	names.append("word_inhibition")

	#parameters.append(pm.max_activity)
	#bounds.append((0, 5))
	#names.append("max_activity")

	#parameters.append(pm.max_attend_width)
	#bounds.append((3, 9))
	#names.append("max_attend_width")

	#parameters.append(pm.min_attend_width)
	#bounds.append((1,3))
	#names.append("min_attend_width")

#	parameters.append(pm.attention_skew)
#	bounds.append((1, 8))
#	names.append("attention_skew")

	#parameters.append(pm.min_overlap)
	#bounds.append((1, 10))
	#names.append("min_overlap")

	#parameters.append(pm.refix_size)
	#bounds.append((0, 2))
	#names.append("refix_size")

#	parameters.append(pm.salience_position)
#	bounds.append((0, 5))
#	names.append("salience_position")

#	parameters.append(pm.sacc_optimal_distance)
#	bounds.append((3, 10))
#	names.append("sacc_optimal_distance")

	#parameters.append(pm.saccErr_scaler)
	#bounds.append((0, 3))
	#names.append("sacc_err_scaler")

	#parameters.append(pm.saccErr_sigma)
	#bounds.append((0, 1))
	#names.append("sacc_err_sigma")

	#parameters.append(pm.saccErr_sigma_scaler)
	#bounds.append((0, 1))
	#names.append("sacc_err_sigma_scaler")

	#parameters.append(pm.mu)
	#bounds.append((1, 10))
	#names.append("mu")

#	parameters.append(pm.sigma)
#	bounds.append((0.5, 8))
#	names.append("sigma")

#	parameters.append(pm.distribution_param)
#	bounds.append((0.5, 5))
#	names.append("distribution_param")

#	parameters.append(pm.wordfreq_p)
#	bounds.append((1,15))
#	names.append("wordfreq_p")

#	parameters.append(pm.wordpred_p)
#	bounds.append((1,15))
#	names.append("wordpred_p")

	return parameters, bounds, names
