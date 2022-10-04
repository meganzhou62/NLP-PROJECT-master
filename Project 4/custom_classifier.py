import dataloader as dataloader
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import random

unk = '<unk>'

def sentiment(content):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(content)
    return (scores['neg'], scores['neu'], scores['pos'], scores['compound'])


def to_feature_vec(s1, s2, s3, s4, ending1, ending2, vectorizor, char_vectorizor):
    story_sentiments = []
    sen = sentiment(s1 + " " + s2 + " " + s3 + " " + s4)
    story_sentiments += [sen[0], sen[1], sen[2], sen[3]]
    ending1_sentiment = sentiment(ending1)
    ending2_sentiment = sentiment(ending2)
    ending1Length = len(word_tokenize(ending1))
    ending2Length = len(word_tokenize(ending2))
    feature_vec = story_sentiments + [ending1_sentiment[0], ending1_sentiment[1], ending1_sentiment[2], ending1_sentiment[3], ending2_sentiment[0], ending2_sentiment[1], ending2_sentiment[2], ending2_sentiment[3], ending1Length, ending2Length]
    story_ngram_vec = vectorizor.transform([s4])
    story_ngram_vec = TfidfTransformer().fit_transform(story_ngram_vec)
    ending1_ngram_vec = vectorizor.transform([ending1])
    ending1_ngram_vec = TfidfTransformer().fit_transform(ending1_ngram_vec)
    ending2_ngram_vec = vectorizor.transform([ending2])
    ending2_ngram_vec = TfidfTransformer().fit_transform(ending2_ngram_vec)
    story_ngram_char_vec = char_vectorizor.transform([s4])
    story_ngram_char_vec = TfidfTransformer().fit_transform(story_ngram_char_vec)
    ending1_ngram_char_vec = char_vectorizor.transform([ending1])
    ending1_ngram_char_vec = TfidfTransformer().fit_transform(ending1_ngram_char_vec)
    ending2_ngram_char_vec = char_vectorizor.transform([ending2])
    ending2_ngram_char_vec = TfidfTransformer().fit_transform(ending2_ngram_char_vec)
    feature_vec = feature_vec + list(story_ngram_vec.toarray()[0]) + list(ending1_ngram_vec.toarray()[0]) + list(ending2_ngram_vec.toarray()[0])
    feature_vec = feature_vec + list(story_ngram_char_vec.toarray()[0]) + list(ending1_ngram_char_vec.toarray()[0]) + list(ending2_ngram_char_vec.toarray()[0])
    return feature_vec

def main():
    train = dataloader.fetch_untokenized("datasets/train.csv")
    trainX = []
    trainY = []
    train_corpus = [s1 + " " + s2 + " " + s3 + " " + s4 + " " + e1 + " " + e2 for s1,s2,s3,s4,e1,e2,_ in train]
    countVectorizer = CountVectorizer(ngram_range=(1,2)).fit(train_corpus)
    charVectorizer = CountVectorizer(analyzer='char', ngram_range=(1,3)).fit(train_corpus)
    sys.stdout.flush()
    print("transforming training data to vectors")
    sys.stdout.flush()
    for s1, s2, s3, s4, ending1, ending2, gold_label in train:
        featureVec = to_feature_vec(s1, s2, s3, s4, ending1, ending2, countVectorizer, charVectorizer)
        trainX.append(featureVec)
        trainY.append(gold_label)
    
    print("training classifier")
    sys.stdout.flush()
    classifier = LogisticRegression()
    classifier.fit(trainX, trainY)
    
    total = 0
    correct = 0
    print("traing accuracy: ")
    sys.stdout.flush()
    for s1, s2, s3, s4, ending1, ending2, gold_label in train:
        featureVec = to_feature_vec(s1, s2, s3, s4, ending1, ending2, countVectorizer, charVectorizer)
        predicted = classifier.predict([featureVec])
        if predicted[0] == gold_label:
            correct += 1
        total += 1
    print(correct / total)

    total = 0
    correct = 0
    valid = dataloader.fetch_untokenized("datasets/dev.csv")
    valid_as_test = dataloader.fetch_test_untokenized("datasets/dev.csv")
    print("validation accuracy: ")
    predictions = []
    for s1, s2, s3, s4, ending1, ending2, gold_label in valid:
        featureVec = to_feature_vec(s1, s2, s3, s4, ending1, ending2, countVectorizer, charVectorizer)
        predicted = classifier.predict([featureVec])
        predictions.append(predicted[0])
        if predicted[0] == gold_label:
            correct += 1
        total += 1
    print(correct / total)
    df = pd.DataFrame({"Id" : [x for (x,_,_,_,_,_,_) in valid_as_test], "Prediction" : predictions, "gold_labels" : [x for (_,_,_,_,_,_,x) in valid]})
    df.to_csv("custom_validation_result.csv")

    print("generating test result")
    test_data = dataloader.fetch_test_untokenized("datasets/test.csv")
    predictions = []
    for _, s1, s2, s3, s4, ending1, ending2 in test_data:
        featureVec = to_feature_vec(s1, s2, s3, s4, ending1, ending2, countVectorizer, charVectorizer)
        predicted = classifier.predict([featureVec])
        predictions.append(predicted[0])
    df = pd.DataFrame({"Id": [x for (x,_,_,_,_,_,_) in test_data], "Prediction": predictions})
    df.to_csv("custom.csv", index=False)


main()