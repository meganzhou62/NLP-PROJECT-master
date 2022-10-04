import csv
from collections import Counter
from numpy import log
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

import numpy as np
import random
import ast

### Training ###

#Set constants
unknow_prob = 0.00000001
lamda = 0.3
k = 0.8

#stores tokenized corpus text
words_training=[]

#stores POS Tags
parts_of_words_training=[]

#stores metaphor tags
labels_training=[]

def get_features(document, pos, index):
    return {
        'word': document[index],
        'word-1': document[index-1] if index - 1 >= 0 else "",
        'word-2': document[index-2] if index - 2 >= 0 else "",
        'word-3': document[index-3] if index - 3 >= 0 else "",
        'word+1': document[index+1] if index + 1 < len(document) else "",
        'word+2': document[index+2] if index + 2 < len(document) else "",
        'word+3': document[index+3] if index + 3 < len(document) else "",
        'pos-1' : pos[index-1] if index - 1 >= 0 else "",
        'pos-2' : pos[index-2] if index - 2 >= 0 else "",
        'pos-3' : pos[index-3] if index - 3 >= 0 else "",
        'pos-1' : pos[index+1] if index + 1 < len(document) else "",
        'pos-2' : pos[index+2] if index + 2 < len(document) else "",
        'pos-3' : pos[index+3] if index + 3 < len(document) else ""
    }



with open('./data_release/train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        words_training.append([w.lower() for w in line[0].split()])
        parts_of_words_training.append(ast.literal_eval(line[1]))
        labels_training.append(ast.literal_eval(line[2]))

#calculate emission probabilities P(Wi | ti), first get counts then normalize.
lexicon_generation = []
for i in range(len(words_training)):
    lexicon_generation += zip(labels_training[i], words_training[i])

all_words = list(set(x for l in words_training for x in l))
all_pos = list(set(x for l in parts_of_words_training for x in l))
num_of_word = len(all_words)
num_of_pos = len(all_pos)
words_dict = {k: v for v, k in enumerate(all_words)}
pos_dict = {k: v for v, k in enumerate(all_pos)}

lexicon_generation_matrix = np.full((2, num_of_word), k)
for bigram in lexicon_generation:
    metaphor, word = bigram
    lexicon_generation_matrix[metaphor][words_dict[word]]+=1

row_sums = lexicon_generation_matrix.sum(axis=1)
lexicon_generation_matrix = lexicon_generation_matrix /row_sums[:, np.newaxis]

# Transform traning set to features and output
X, Y = [], []
for doc_index in range(len(words_training)):
    for word_index in range(len(words_training[doc_index])):
        X.append(get_features(words_training[doc_index], parts_of_words_training[doc_index], word_index))
        Y.append(labels_training[doc_index][word_index])

classifier = Pipeline([
    ('vectorizer', DictVectorizer()),
    ('classifier', MultinomialNB(alpha=k))
])

classifier.fit(X, Y)

### Training Done ###

def get_prediction(classifier, lexicon_generation_matrix, words_dict, words, pos):
    hmm_matrix = np.zeros((2, len(words)))
    back_pointer = np.zeros((2, len(words)))
    #Generate the first column for HMM
    firstword = words[0]
    #Use probability from classifier
    print(words[:5])

    for i in range(2):
        if firstword in words_dict:
            hmm_matrix[i][0] = lamda * classifier.predict_log_proba(get_features(words, pos, 0))[0][i] + log(lexicon_generation_matrix[i][words_dict[firstword]])
        else:
            hmm_matrix[i][0] = lamda * classifier.predict_log_proba(get_features(words, pos, 0))[0][i] + log(unknow_prob)
        back_pointer[i][0] = 0

    # Run Viterbi
    for j in range(1, len(words)):
        for i in range(2):
            compare = {}
            for tag in range(2):
                if words[j] in words_dict:
                    compare[tag] = hmm_matrix[tag][j-1] + lamda * classifier.predict_log_proba(get_features(words, pos, j))[0][i] + log(lexicon_generation_matrix[i][words_dict[words[j]]])
                else:
                    compare[tag] = hmm_matrix[tag][j-1] + lamda * classifier.predict_log_proba(get_features(words, pos, j))[0][i] + log(unknow_prob)
            back_pointer[i][j] = max(compare, key=compare.get)
            hmm_matrix[i][j] = compare[back_pointer[i][j]]

    # identify sequence
    t = np.zeros(len(words))
    t[len(words) - 1] = np.argmax(hmm_matrix[:, len(words) - 1], axis=0)

    for i in reversed(range(len(words) - 1)):
        t[i] = back_pointer[int(t[i + 1])][i + 1]

    return t.astype(int).tolist()

### Training Done ###

def get_prediction(classifier, lexicon_generation_matrix, words_dict, words, pos):
    hmm_matrix = np.zeros((2, len(words)))
    back_pointer = np.zeros((2, len(words)))
    #Generate the first column for HMM
    firstword = words[0]
    #Use probability from classifier

    for i in range(2):
        if firstword in words_dict:
            hmm_matrix[i][0] = lamda * classifier.predict_log_proba(get_features(words, pos, 0))[0][i] + log(lexicon_generation_matrix[i][words_dict[firstword]])
        else:
            hmm_matrix[i][0] = lamda * classifier.predict_log_proba(get_features(words, pos, 0))[0][i] + log(unknow_prob)
        back_pointer[i][0] = 0

    # Run Viterbi
    for j in range(1, len(words)):
        for i in range(2):
            compare = {}
            for tag in range(2):
                if words[j] in words_dict:
                    compare[tag] = hmm_matrix[tag][j-1] + lamda * classifier.predict_log_proba(get_features(words, pos, j))[0][i] + log(lexicon_generation_matrix[i][words_dict[words[j]]])
                else:
                    compare[tag] = hmm_matrix[tag][j-1] + lamda * classifier.predict_log_proba(get_features(words, pos, j))[0][i] + log(unknow_prob)
            back_pointer[i][j] = max(compare, key=compare.get)
            hmm_matrix[i][j] = compare[back_pointer[i][j]]

    # identify sequence
    t = np.zeros(len(words))
    t[len(words) - 1] = np.argmax(hmm_matrix[:, len(words) - 1], axis=0)

    for i in reversed(range(len(words) - 1)):
        t[i] = back_pointer[int(t[i + 1])][i + 1]

    return t.astype(int).tolist()


### Evluate with validation set ###
def eval():
    predictions = []
    gold_labels = []
    with open('./data_release/val.csv', encoding='latin-1') as f:
        lines = csv.reader(f)

        next(lines)
        for line in lines:
            pos = ast.literal_eval(line[1])
            label_seq = ast.literal_eval(line[2])
            words = [w.lower() for w in line[0].split()]

            predictions.append( get_prediction(classifier, lexicon_generation_matrix, words_dict, words, pos))

            for i in range(len(words)):
                gold_labels.append(label_seq[i])




    return predictions

pred_nb = eval()

import csv
from collections import Counter
from numpy import log

import numpy as np
import random
import ast
import pandas as pd

### Training ###

# Set constants
unknow_prob = 0.00000001
lamda = 0.5
k = 0.1

# stores corpus text
words_training = []

# stores POS Tags
parts_of_words_training = []

# stores metaphor tags
labels_training = []

with open('./data_release/train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        words_training.append([w.lower() for w in line[0].split()])
        parts_of_words_training.append(ast.literal_eval(line[1]))
        labels_training.append(ast.literal_eval(line[2]))

# get transitional counts for metaphor tags
bigrams_metaphor = [b for l in labels_training for b in zip(l[:-1], l[1:])]
transitional_matrix = np.full((2, 2), k)

for bigram in bigrams_metaphor:
    first, second = bigram
    transitional_matrix[first][second] += 1
transitional_matrix = np.array(transitional_matrix)

# Normalize counts and turn into probability
row_sums = transitional_matrix.sum(axis=1)
transitional_matrix = transitional_matrix / row_sums[:, np.newaxis]

# generate the "start" probabilities, used in the first step of viterbi.
start_probability = np.ones(2)
for txt in labels_training:
    start_probability[txt[0]] += 1
sum = sum(start_probability)
start_probability = start_probability / sum

# calculate emission probabilities P(Wi | ti), first get counts then normalize.
lexicon_generation = []
for i in range(len(words_training)):
    lexicon_generation += zip(labels_training[i], words_training[i])

all_words = list(set(x for l in words_training for x in l))
num_of_word = len(all_words)
words_dict = {k: v for v, k in enumerate(all_words)}

lexicon_generation_matrix = np.full((2, num_of_word), k)
for bigram in lexicon_generation:
    metaphor, word = bigram
    lexicon_generation_matrix[metaphor][words_dict[word]] += 1

row_sums = lexicon_generation_matrix.sum(axis=1)
lexicon_generation_matrix = lexicon_generation_matrix / row_sums[:, np.newaxis]


### Training Done ###

# Returns
def get_prediction(transitional_matrix, lexicon_generation_matrix, words_dict, words):
    hmm_matrix = np.zeros((2, len(words)))
    back_pointer = np.zeros((2, len(words)))
    # Generate the first column for HMM
    firstword = words[0]
    # Use calculated probability if word seen, otherwise use preset probablity for unknown words
    for i in range(2):
        if firstword in words_dict:
            hmm_matrix[i][0] = log(start_probability[i]) * lamda + log(
                lexicon_generation_matrix[i][words_dict[firstword]])
        else:
            hmm_matrix[i][0] = log(start_probability[i]) * lamda + log(unknow_prob)

        back_pointer[i][0] = 0
    # Run Viterbi
    for j in range(1, len(words)):
        for i in range(2):
            compare = {}
            for tag in range(2):
                if words[j] in words_dict:
                    compare[tag] = hmm_matrix[tag][j - 1] + log(transitional_matrix[tag][i]) * lamda + \
                                   log(lexicon_generation_matrix[i][words_dict[words[j]]])
                else:
                    compare[tag] = hmm_matrix[tag][j - 1] + log(transitional_matrix[tag][i]) * lamda + log(unknow_prob)

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

        predictions.append( get_prediction(transitional_matrix, lexicon_generation_matrix, words_dict, words))

        gold_labels.append(label_seq)



df = pd.DataFrame(range(len(gold_labels)))
df = df.join(pd.DataFrame({"true": gold_labels, "prediction by hmm": predictions, "prediction with our model": pred_nb, 'sentence': allsentences}))


df.to_csv('error analysis.csv', index=False)

