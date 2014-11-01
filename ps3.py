"""
Team Challenge: Predicting Turn Types
Authors: Tong, Will, and Ryan
"""

import os
import re
import sys
import csv
import argparse
from pprint import pprint
from time import time

import logging
from operator import itemgetter
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

def read_csv(path):
    output = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            output.append(row)
    return output

def read_dir_sk(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    Returns an array of dicts, array of qa labels, and an array of em labels
    """
    X = []
    y_qa = []
    y_em = []
    for root, subdirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                a_file_path = os.path.join(root, f)
                csv = read_csv(a_file_path)

                for row in csv:
                    y_qa.append(row[3])
                    y_em.append(row[4])
                    # remove asterisk *
                    # remove entire repair turns <>
                    # remove brackets around words []
                    text = re.sub(r'(<|>|\[|\]|\*)', '', row[5])
                    X.append(text)
    return X, y_qa, y_em

def read_dir_dict(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    Returns an array of dicts
    """
    output = []
    for root, subdirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                a_file_path = os.path.join(root, f)
                csv = read_csv(a_file_path)

                for row in csv:
                    example = { "subjectID": row[0],
                                "imageID": row[1],
                                "questionID": row[2],
                                "Q/A": row[3],
                                "E/M": row[4],
                                "text": row[5] }
                    output.append(example)
    return output

def q_features(data):
    questionWords = ["who", "what", "where", "when", "why", "[why]", "how", "and"]
    ansWords = ["sp", "well", "uh", "hm", "it", "mm"]

    totalCount = 0
    qCount = 0
    aCount = 0
    for d in data:
        totalCount += 1
        for q in questionWords:
            if d["text"].startswith(q):
                qCount += 1
                break
        for a in ansWords:
            if d["text"].startswith(a):
                aCount += 1
                break

    print("Total: ", totalCount, "Q: ", qCount, "A: ", aCount)
    print("Ratio: ", (qCount + aCount)/totalCount)

def encode_labels(y):
    """
    Takes a list of labels and converts them to numpy compatible classes
    Returns (the converted labels, and the encoder)
    """
    le = LabelEncoder()
    y = le.fit_transform(y)

    # example of how to get the labels back:
    # print le.inverse_transform([1, 0, 1, 0, 1, 0, 1, 0])
    return y, le

def run_pipeline(data, targets):
    pipe = Pipeline([
        ("vect", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])

    params = {
        #"vect__max_df": (0.5, 0.75, 1.0),
        #"vect__max_features": (None, 5000, 10000, 50000),
        "vect__use_idf": (True, False),
        #"vect__analyzer": ("word", "char"),
        "vect__ngram_range": ((1,2), (1,3), (1,1)),
        "vect__norm": ("l1", "l2"),
        "clf__alpha": (0.001, 0.00001, 0.000001)
    }

    grid_search= GridSearchCV(pipe, params, cv=KFold(len(targets), 11), verbose=1)

    print("Performing grid search...")
    print("pipe:", [name for name, _ in pipe.steps])
    print("params:")
    pprint(params)
    t0 = time()
    grid_search.fit(data, targets)
    print("done in %0.3fs" % (time() - t0))
    print

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best params set:")
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_params[param_name]))

    report(grid_search.grid_scores_)
    return grid_search.best_estimator_

def report(grid_scores, n_top=10):
    """
    Helper function to report score performance
    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("{2}. Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores),
              i + 1 ))
        print("Parameters: {0}".format(score.parameters))
        print("")

def main(args):
    # Our features and two sets of labels
    corpus, y_qa, y_em = read_dir_sk(args.data)

    y_qa, le_qa = encode_labels(y_qa)
    y_em, le_em = encode_labels(y_em)

    print("--- Q/A ---")
    best_qa_clf = run_pipeline(corpus, y_qa)
    # leave out the first dataset to simulate test data performance
    #best_qa_clf = run_pipeline(corpus[40:], y_qa[40:])
    #print("Performance on the left out dataset: {0}".format(
        #best_qa_clf.score(corpus[:40], y_qa[:40])))

    print
    print("--- E/M ---")
    best_em_clf = run_pipeline(corpus, y_em)
    #best_em_clf = run_pipeline(corpus[40:], y_em[40:])
    #print("Performance on the left out dataset: {0}".format(
        #best_em_clf.score(corpus[:40], y_em[:40])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If no explicit training data, algorithm splits it itself
    parser.add_argument( "-d", "--data", help="pass a folder path to the data")

    # Otherwise
    parser.add_argument( "-t", "--train", help="pass a folder path to the training data")
    parser.add_argument( "-s", "--test", help="pass a folder path to the testing data")

    args = parser.parse_args()
    main(args)
