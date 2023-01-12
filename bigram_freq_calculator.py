path = r'C:\Users\Josh\Desktop\Josh work\Experiments\position priming\dlp-items.txt' # lexicon

with open(path) as file:
    all_words = file.readlines()
for i in range(0,len(all_words)):
    all_words[i] = all_words[i].split('\t')

real_words = []

for i in range(0,len(all_words)):
    if all_words[i][1] == 'W':
        real_words.append(all_words[i][0])

bigrams = {}
bigram_counter = 0

for i in range(0,len(real_words)):
    for letter1 in range(0,len(real_words[i])-1):
        for letter2 in range(1,len(real_words[i])):
            if letter1 < letter2:
                bigram = real_words[i][letter1]+real_words[i][letter2]
                if bigram in bigrams:
                    bigrams[bigram] += 1
                else:
                    bigrams[bigram] = 0
                bigram_counter += 1


for key in bigrams:
    bigrams[key] = float(bigrams[key])/bigram_counter   # this gives the proportions of all bigrams in the Dutch language

#-----------------------------------------------------------------------------------------------------------------



path = r'C:\Users\Josh\Desktop\Josh work\Experiments\position priming\__pool__\FLLD_word.txt' # our stimuli

with open(path) as file:
    stimuli = file.readlines()
    
same_pos_primes = []
diff_pos_primes = []    
for i in range(0,len(stimuli)):
    same_pos_primes.append(stimuli[i].split('..')[1])
    diff_pos_primes.append(stimuli[i].split('..')[2])
    
same_pos_prime_bigram_freq = []
diff_pos_prime_bigram_freq = []

for i in range(0,len(same_pos_primes)):
    same_pos_prime_bigrams = []
    diff_pos_prime_bigrams = []
    for letter1 in range(0,len(same_pos_primes[i])-1):
        for letter2 in range(1,len(same_pos_primes[i])):
            if letter1 < letter2:
                bigram = same_pos_primes[i][letter1]+same_pos_primes[i][letter2]
                if bigram in bigrams:
                    same_pos_prime_bigrams.append(bigrams[bigram])
                else:
                    same_pos_prime_bigrams.append(0)
                    
                bigram = diff_pos_primes[i][letter1]+diff_pos_primes[i][letter2]
                if bigram in bigrams:
                    diff_pos_prime_bigrams.append(bigrams[bigram])
                else:
                    diff_pos_prime_bigrams.append(0)
    same_pos_prime_bigram_freq.append(float(sum(same_pos_prime_bigrams))/len(same_pos_prime_bigrams))
    diff_pos_prime_bigram_freq.append(float(sum(diff_pos_prime_bigrams))/len(diff_pos_prime_bigrams))

same_pos_prime_bigram_freq = float(sum(same_pos_prime_bigram_freq))/len(same_pos_prime_bigram_freq)
diff_pos_prime_bigram_freq = float(sum(diff_pos_prime_bigram_freq))/len(diff_pos_prime_bigram_freq)
    
    
    
    
    
    
    