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

            predictions += get_prediction(classifier, lexicon_generation_matrix, words_dict, words, pos)

            for i in range(len(words)):
                gold_labels.append(label_seq[i])

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


    print('P, R, F1, Acc.')
    print(precision, recall, met_f1, accuracy)
    print('lamda, k')
    print(lamda, k)


def test():
    predictions = []
    with open('./data_release/test_no_label.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            pos = ast.literal_eval(line[1])
            words = [w.lower() for w in line[0].split()]
            predictions += get_prediction(classifier, lexicon_generation_matrix, words_dict, words, pos)
    indices = [x + 1 for x in list(range(len(predictions)))]
    df = pd.DataFrame({"idx": indices, "label": predictions})
    df.to_csv('lamda-%s-k=%f-3windowNB.csv' % (lamda, k), index=False)


eval()
