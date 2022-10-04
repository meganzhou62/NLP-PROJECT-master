import csv
from collections import Counter
from numpy import log

import numpy as np
import random
import ast
import pandas as pd
### Training ###

#Set constants
unknow_prob = 0.00000001
lamda = 0.5
k = 0.1

#stores corpus text
words_training=[]

#stores POS Tags
parts_of_words_training=[]

#stores metaphor tags
labels_training=[]

with open('./data_release/train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        words_training.append([w.lower() for w in line[0].split()])
        parts_of_words_training.append(ast.literal_eval(line[1]))
        labels_training.append(ast.literal_eval(line[2]))

#get transitional counts for metaphor tags
bigrams_metaphor = [b for l in labels_training for b in zip(l[:-1], l[1:])]
transitional_matrix = np.full((2, 2),k)

for bigram in bigrams_metaphor:
    first, second = bigram
    transitional_matrix[first][second] += 1
transitional_matrix = np.array(transitional_matrix)

#Normalize counts and turn into probability
row_sums = transitional_matrix.sum(axis=1)
transitional_matrix = transitional_matrix /row_sums[:, np.newaxis]

# generate the "start" probabilities, used in the first step of viterbi.
start_probability = np.ones(2)
for txt in labels_training:
    start_probability[txt[0]] += 1
sum = sum(start_probability)
start_probability = start_probability/sum

#calculate emission probabilities P(Wi | ti), first get counts then normalize.
lexicon_generation = []
for i in range(len(words_training)):
    lexicon_generation += zip(labels_training[i], words_training[i])

all_words = list(set(x for l in words_training for x in l))
num_of_word = len(all_words)
words_dict = {k: v for v, k in enumerate(all_words)}

lexicon_generation_matrix = np.full((2, num_of_word), k)
for bigram in lexicon_generation:
    metaphor, word = bigram
    lexicon_generation_matrix[metaphor][words_dict[word]]+=1

row_sums = lexicon_generation_matrix.sum(axis=1)
lexicon_generation_matrix = lexicon_generation_matrix /row_sums[:, np.newaxis]

### Training Done ###

#Returns 
def get_prediction(transitional_matrix, lexicon_generation_matrix, words_dict, words):
    hmm_matrix = np.zeros((2, len(words)))
    back_pointer = np.zeros((2, len(words)))
    #Generate the first column for HMM
    firstword = words[0]
    #Use calculated probability if word seen, otherwise use preset probablity for unknown words
    for i in range(2):
        if firstword in words_dict:
            hmm_matrix[i][0] = log(start_probability[i]) * lamda + log(lexicon_generation_matrix[i][words_dict[firstword]])
        else:
            hmm_matrix[i][0] = log(start_probability[i]) * lamda + log(unknow_prob)
        
        back_pointer[i][0] = 0
    # Run Viterbi
    for j in range(1, len(words)):
        for i in range(2):
            compare = {}
            for tag in range(2):
                if words[j] in words_dict:
                    compare[tag] = hmm_matrix[tag][j-1] + log(transitional_matrix[tag][i]) * lamda + \
                                   log(lexicon_generation_matrix[i][words_dict[words[j]]])
                else:
                    compare[tag] = hmm_matrix[tag][j-1] + log(transitional_matrix[tag][i]) * lamda + log(unknow_prob)

            back_pointer[i][j] = max(compare, key=compare.get)
            hmm_matrix[i][j] = compare[back_pointer[i][j]]
    # identify sequence
    t = np.zeros(len(words))
    t[len(words) - 1] = np.argmax(hmm_matrix[:, len(words) - 1], axis=0)

    for i in reversed(range(len(words) - 1)):
        t[i] = back_pointer[int(t[i + 1])][i + 1]

    return t.astype(int).tolist()


### Evluate with validation set ###
allsentences = []
predictions = []
gold_labels = []
with open('./data_release/val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)

    next(lines)
    for line in lines:

        label_seq = ast.literal_eval(line[2])
        words = [w.lower() for w in line[0].split()]
        allsentences.append(words)


        predictions += get_prediction(transitional_matrix, lexicon_generation_matrix, words_dict, words)

        gold_labels.append(label_seq)
print(gold_labels)
assert(len(predictions) == len(gold_labels))
total_examples = len(predictions)


num_correct = 0
confusion_matrix = np.zeros((2, 2))
for i in range(total_examples):
    if predictions[i] == gold_labels[i]:
        num_correct += 1
    confusion_matrix[predictions[i], gold_labels[i]] += 1

assert(num_correct == confusion_matrix[0, 0] + confusion_matrix[1, 1])
accuracy = 100 * num_correct / total_examples
precision = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[1])
recall = 100 * confusion_matrix[1, 1] / np.sum(confusion_matrix[:, 1])
met_f1 = 2 * precision * recall / (precision + recall)


df = pd.DataFrame(gold_labels)
df = df.join(pd.DataFrame({"prediction":predictions}))
#df.to_csv('hmmsimplepredictionlamda-%s-k=%f.csv'% (lamda, k), index=False)

print('P, R, F1, Acc.')
print(precision, recall, met_f1, accuracy)
print(unknow_prob,lamda, k )