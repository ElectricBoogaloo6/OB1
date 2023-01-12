import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pickle
from reading_common import get_stimulus_text_from_file

#function to get all indices of a value from a list
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

def draw_boxplot(classes_array,values_array,ax):
	unique_classes = np.unique(fixation_position_lengths)
		
	boxplot_values_list = []
	for class_name in unique_classes:
		indexes = all_indices(class_name,classes_array)
		values = []
		for value_index in indexes:
			values.append(values_array[value_index])

		boxplot_values_list.append(values)

	plt.boxplot(boxplot_values_list)
	plt.xticks(unique_classes)
	plt.ylim(max( min(values_array) *0.8, 0), max(values_array)*1.25  )

input_text_filename = "texts/POS.txt"
textfile=get_stimulus_text_from_file(input_text_filename)

input_text_filename = "texts/descartes.txt"
textfile2=get_stimulus_text_from_file(input_text_filename)


individual_words = []
textsplitbyspace = textfile.split(" ")
for word in textsplitbyspace:
    if word.strip()!="":
        individual_words.append(word.strip())

word_lengths = []
for word in individual_words:
	word_lengths.append(len(word))

individual_words = []
textsplitbyspace = textfile2.split(" ")
for word in textsplitbyspace:
    if word.strip()!="":
        individual_words.append(word.strip())

word_lengths2 = []
for word in individual_words:
	word_lengths2.append(len(word))

minx = 0
maxx=18
binwidth =3

ax = plt.subplot(121)
ax.set_title("english word length frequencies")
ax.hist(word_lengths,bins=np.arange(0,20,binwidth),normed=1)
ax.axis([minx, maxx, 0, 0.25])

ax = plt.subplot(122)
ax.set_title("dutch word length frequencies")
ax.axis([minx, maxx, 0, 0.25])
ax.hist(word_lengths2,bins=np.arange(0,20,binwidth),normed=1)


plt.show()