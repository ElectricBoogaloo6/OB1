### Note December 2023
German affix splitter has been implemented, more details on that can be found: https://github.com/ElectricBoogaloo6/German_affix_splitter
As the focus of Beyersmann (2020) study was on the nonwords (with combinations of stem/non-stem and suffix/non-suffix) and real complex/simple words for comparison, the affix splitter was implemented along with lexicon and the stimuli from the 2020 study. The plots and results of this can be found in the OB1_analysis folder - OB1_taskperformance_german_nonwords.ipynb.

### Note august 2022

The slot-matching mechanism written by Martijn has been merged. 

Additionally, the internal state of OB1 is recorded for every time step. This data can be consulted in the logs/ folder.


### Note sept 2021 (MM)
This code based on Gina's handover version of April 2021, with Noor Seidel's code for simulating expts integrated in it. 
The code for experiments works well, that for reading text works but with hacks that are difficult to follow (must be rewritten).

# OB1 reader

OB1 is a reading-model that simulates the cognitive processes behind reading. 
For more information about the theory behind OB1 and how it works see: https://www.ncbi.nlm.nih.gov/pubmed/30080066

## Running the code
First, clone this repo and install the requirements with '''git install -r requirements.txt''' (make sure you cd into the OB1 folder)

Then, set your parameters in parameters.py (which task to run, etc.) and run the code by running main.py.

The code can be used for different purposes (code file mentioned are explained below). 

## Reading a text or running an experiment

In order to run a text reading or an experiment, one should set "run_exp" and "analyze_results" in *parameters.py* to True and "optimize" to False. 

### Text reading
To run the "normal" text reading task (Which means reading the input text-file once and comparing the results to an eye-tracking 
experiment), set task_to_run in *parameters.py* to "PSCall". In the standard version it reads a german text and uses word frequency as well as
word predictability (cloze probability) to recognize words presented in its visual field.

### Experiment 
To run an experiment, set task_to_run to the task in question. Can be :
"Flanker", from Snell et al (2019, Neuropsychologia), 

"Sentence", a reading experiment from Wen et al. (2019, Cognition), 

"EmbeddedWords", a priming task from Beyersmann et al. (2016, Psychonomic Society),
For this task, a system for processing affixes has been implemented, to account for the priming results found by i.e. doi:10.3758/s13423-015-0927-z. In the present state, word pairs of the complex words and their stem (i.e. weaken - weak), are detected and their inhibition is set to 0. That means that the word pairs dont inhibit each other, which explains why WEAKEN primes WEAK, but CASHEW does not prime CASH (no affix).
The affix system can be turned on or off by setting *affix_system* in parameters.py to True or False.
At the moment, the affix system is fully functional for english and french, which means there exists a pickle file containing affix frequencies for those lamguages, which can be used to anaylse affix-effects in simulations. 

"Classification" and " Transposed ".
These last two experiments make use of the grammar predictability of words and their POS. The POS and grammar prob have been implemented only for these two tasks (english and dutch). The grammar prob code of Beatriz is functional, but is a little disorganized. it works, but if needed to extend for PSC or other tasks, should contact Beatriz.

The simulated experiment data is stored in pickled files called alldata_Flanker.pkl, etc. that can be read by Jupyter Notebooks written by Noor Seidel.
The Notebooks expect the pickled files in "...\Results". Run OB1_taskperformance (stiutated in the OB1_analysis folder) before the other two (which compute ERPs and simulated ERPs, one for each task).

## Parameter-tuning 

In this version the model is executed multiple times in order to find the set of parameters that enables the model to 
read in a way that is similar to how a human would read the text. The optimization is done by using the *L-BFGS-B* 
optimization method from *scipy*.
For parameter-tuning define the parameters you wish to change and their bounds in *get_parameters.py*. Then go to 
*reading_simulation.py* where you have to unpack these values again based on the order in which they have been packed.
 
Next go to *parameters.py* and change "optimize" to True. Don't forget to set "run_exp" as well as "analyze_results" to
False if you want to **just optimize**.
  
The parameters are saved if they are better than the parameters from the previous iteration. They are saved
as a text file named after the tuning measure and the distance between experiment and simulation. 


## adding a new experiment 
When implementing a new task, head to parameters.py, input it in the list of possible tasks, and set its attributes. Add a CSV with stimuli in the /Stimuli map. 
Be careful to match the column structure of the other CSV's.

## adding a new language
To add a new language there has to be the plain text as input data for the reading simulation (file location defined in main.py, see *PSC_ALL.txt* as an example for the format), 
a lexicon (see word_freq.pkl as an example, file location defined in function "get_freq..." in read_saccade_data) as well as the preprocessed eyetracking-data recorded during an experiment
 where participants had to read the text that is presented to OB1. For an example of the input data derived from an eye-tracking experiment see the table stored in *Fixation_durations_german.pkl*.
The right files are now only available fo German/Potsdam corpus.

## files in the directory
The following files are the most important: 

### parameters.py
This is the most important function for controlling the behavior of *main.py*. Here the user can specify which parts of the programm should be run and also set the initial parameters when tuning. 
Furthermore the user can define which measures are used as error-function for the tuning process. 
This is also where the specifics of every task are specified.

For plotting inhibition and activation values during simulation, set plotting=True in parameters.py (useful for debugging and insight in to the model at simulation time).

It mainly consists of multiple important functions: 
* "return_global_params()" - This function sets the task to run along with parameters such as "run_exp", "analyze_results", and "optimize". These parameters are in charge of setting the experiment run and analyzing results.
* "return_attributes(task_to_run)" - Returns an instance of the TaskAttribute class. This class sets the atributes of the task, like stimuli, language and the number of stimulus cycles. Importantly, this function takes in an argument "task_to_run" which determines which task's attributes will be returned. For example: when the task is 'EmbeddedWords_German', the function reads a CSV file of the German stimuli, assigns it to the stim attribute of the TaskAttributes class, assigns 'German' to the language attribute and returns an instance of TaskAttributes with the associated attributes.
* "return_task_params" - sets specific parameters for the task that is chosen to run, such as the time cycle of the task and other parameters such as bigram to word excitation, word inhibition, and attention. These parameters are set for experimentation and are established by the task_attributes object.

### main.py
In this file the main program flow is defined. In case of text reading it has calls to the reading_function, which simulates the actual reading, as imported from *reading_simulation.py*, the analyze function as imported from *analyse_data_pandas.py* and the optimize function, 
which is scipy's *L-BFGS-B* optimizing method. 
The function called by this optimizing method is a wrapper that takes the parameters called in *parameters.py* and feeds them to the reading simulation. The optimize function makes use of a slightly adapted version of the analyzing function that can be found in *get_scores.py*.
In case of experiment running, it calls simulate_experiments. Functionally, it also records the start time of the simulation, and uses logging to create a new log file with the current date and time. It also uses pickle module to save the results of the simulation to a file. The script also check the different parameters and make sure they are in the correct form. It also prints the time it took for the script to run.



### reading_simulation.py
This file embodies the heart of the whole programm, the reading simulation. Here the input text is fed into the visuo-spatial representation which is activating bigramm-matrices, which in turn are activating words that are recognized. 
The resulting (correctly or incorrectly) recognized words are saved in **all_data**, together with a set of descriptive variables. 
At the end of the simulation this data-representation of the reading process is saved as a pickle file ( *all_data_INPUT_LANGUAGE.pkl* ) for analysis in a later stage together with all **unrecognized_words** ( *unrecognized_INPUT_LANGUAGE.pkl* ).
reading_simulation_BT.py is meant for the boundary task (BT). This sim has not been updated for a long while. If BT sims must be run, better use reading_simulation as basis and take whatever needed to do boundary from the BT file.
**TODO:** remake the code to the image of simulate_experiments.py
**NOTE:** deprecated

### simulate_experiments.py
This file has the code for simulating concrete experiments, currently a flanker expt from Snell et al (2019, Neuropsychologia), and a sentence reading experiment from Wen et al. 
(2019, Cognition), or an Embedded Words task (Beyersmann et al. 2016). In the flanker expt, a target word is presented either alone or surrounded by two flanker words on the screen. 

It starts off with importing the stimulus list, and building the lexicon and inter-word inhibition matrix.
The inhibition matrix calculation is the most expensive step in the model. Therefore, the code now first checks if the last run was with the same parameters relevant for inhibition, and if so, uses the previous inhibition matrix, thereby saving redundant computation.
In parallel, it also iterates through every word once instead of twice. 

It then enters the loop, where the stimulus is presented and the word activations are updated at every iteration.

A hit is scored if the target word is recognized in timely fashion. In the sentence reading expt, a sentence (either correct or scrambled) is presented, and the reader has to read aloud a word indicated by a post cue. 
In the Embedded Words task, a prime is presented for 50ms, followed by a target. The user presses a button when the target word is recognized. The prime can be truly suffixed, pseudo suffixed or non-sufixed. head to Beyersamnn et al. (2020) for more info.
In the simulation, we count a hit when the cued word was recognized on time. In both experiments, total activity of word units is added up as substrate of the N400. This is then compared with experimental data using Jupyter Notebooks (see below). 

### reading_common.py
Helper-functions for the reading and experiment simulation 

### read_saccade_data.py
This file provides functions to read in the eye-tracking data collected during the experiment where participants had to read the same text that is presented to the OB1-reader. The functions for reading lexicons, word frequencies and cloze data are also here.

### analyse_data_plot.py / analyse_data_plot_qualitative.py
In this files the result of a single experiment is analyzed and plots as seen in the publication are produced.

### analyse_data_transformation / analyse_data_pandas.py
These files are providing various functions to analyze the data used in *analyse_data_pandas.py*. Importantly, the parameters for analysis are set using the "return_params()" function. The scripts utilises the "get_results()" function which takes in 3 inputs, the input text file name, all data file, and unrecognized words file. It reads the text file and splits it by spaces to get a list of words. Then, it appends all non-empty words to a list called "individual_words" and creates a dataframe from it. It also has specific functionality, for example: if the language is set to German, it uses the "get_freq_and_pred()" function to get frequency and prediction values for the words.

### freq_pred_files.py
This is used for loading in frequency and predictability list of words for a specific task. It then creates a pickle file containing the words, their frequencies and predictabilities, and stores this file in the /Data folder. It uses the "return_params" function from the "parameters.py" script to get the task and language that needs to be used. It uses pandas to read and work with CSV files, and pickle to save the created file.

### read_saccade_data.py
This file reads the data from the file "Fixation_durations_complete.txt" and converts tihs data to a dataframe.The data contains information regarding fixation durations and saccade distances. The script provides several functions that can be used to analyze the data, such as counting the number of saccades at different distances, and calculating the proportion of words that were skipped over in the experiment. "get_sacc_distance()" function counts the number of saccades and gets saccade distances. "wordskiplist" contains the proportion of words that were skipped over. Finally, "get_saccade_durations()" calculates the duration of the saccade at a certain position, "get_saccade_type_probabilities()" function gathers statistical information based on the saccades data, such as refixations (nfp). 


## Concepts: 
* Saccade distances: Saccade distances refer to the distance between two consecutive saccades. Saccades are rapid eye movements that occur when a person shifts their gaze from one point to another (https://journals.sagepub.com/doi/10.1177/0301006616657097). The distance between saccades in combination with its duration can be a useful measure of visual attention and cognitive processing, as well as a marker of reading proficiency.