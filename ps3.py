"""
Team Challenge: Predicting Turn Types
Authors: Tong, Will, and Ryan
"""

import os, re, sys, csv, argparse, logging
from pprint import pprint
from time import time

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics

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

def grid_search_pipeline(pipe, params, cv, data, targets):
    """
    For a given pipeline, parameters, cross validator, dataset, and target labels
    Run a grid search using the given pipeline and cross validator over the set
    of parameters, printing the progress as it goes.
    Returns the results
    """

    grid_search = GridSearchCV(pipe, params, cv=cv, verbose=1)

    print("Performing grid search...")
    print("pipe:", [name for name, _ in pipe.steps])
    print("params:")
    pprint(params)
    print
    t0 = time()
    grid_search.fit(data, targets)
    print("done in %0.3fs" % (time() - t0))

    return grid_search

def tfidf_mnb_pipeline(data, targets, num_images=11):
    """
    A simple pipeline using tfidf vectorizer, multinomial naive bayes, and KFold
    cross validation such that on each run one image file is left out (set
    num_images differently if working on a smaller training set)
    Returns the grid_search results, pipeline, and parameters used
    """
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

    cv = KFold(len(targets), num_images)

    grid_search = grid_search_pipeline(pipe, params, cv, data, targets)

    #return grid_search.best_estimator_
    return grid_search, pipe, params

def report_grid_search(grid_search, pipe, params):
    """
    Report the results of a given grid search
    """
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best params set:")
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_params[param_name]))

    report_grid_scores(grid_search.grid_scores_)

def report_grid_scores(grid_scores, n_top=10):
    """
    Helper function to report score performance of the top n classifier / params
    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("{2}. Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores),
              i + 1 ))
        print("Parameters: {0}".format(score.parameters))
        print("")

def get_metrics(y_test_list, y_pred_list):
    y_true, _ = encode_labels(y_test_list)
    print(y_true)

    y_pred, _ = encode_labels(y_pred_list)
    print(y_pred)
    #for val in y_predfile:
        #val = float(val)
        #if val > 0.0:
            #y_pred.append(1)
        #else:
            #y_pred.append(-1)

    # report the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print 'Confusion Matrix:\n' + str(cm)

    # plot confusion matrix in color in a separate window
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # report the classification metrics
    print( classification_report(y_true, y_pred) )

    # compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print("Area under the ROC curve: %f" % roc_auc)

    # plot ROC curve
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def main(args):
    if (args.data):
        # Our features and two sets of labels
        X, y_qa, y_em = read_dir_sk(args.data)

        y_qa, le_qa = encode_labels(y_qa)
        y_em, le_em = encode_labels(y_em)

        print("--- Q/A ---")
        qa_grid_search, qa_pipe, qa_params = tfidf_mnb_pipeline(X, y_qa)
        report_grid_search(qa_grid_search, qa_pipe, qa_params)
        best_qa_clf = qa_grid_search.best_estimator_

        print
        print("--- E/M ---")
        em_grid_search, em_pipe, em_params = tfidf_mnb_pipeline(X, y_em)
        report_grid_search(em_grid_search, em_pipe, em_params)
        best_em_clf = em_grid_search.best_estimator_

    elif (args.test and args.train):
        train_X, train_y_qa, train_y_em = read_dir_sk(args.train)
        test_X, test_y_qa, test_y_em = read_dir_sk(args.test)

        # how many documents are in the training set?
        len_img_train = int(float(len(train_y_qa))/40.0)
        print(len_img_train)

        print("--- Q/A ---")
        qa_grid_search, qa_pipe, qa_params = tfidf_mnb_pipeline(train_X,
                                                                train_y_qa,
                                                                len_img_train)
        report_grid_search(qa_grid_search, qa_pipe, qa_params)
        best_qa_clf = qa_grid_search.best_estimator_

        print("Q/A performance on the left out dataset: {0}".format(
            best_qa_clf.score(test_X, test_y_qa)))
        qa_predictions = best_qa_clf.predict(test_X)
        print(qa_predictions)

        get_metrics(test_y_qa, qa_predictions)

        print
        print("--- E/M ---")
        em_grid_search, em_pipe, em_params = tfidf_mnb_pipeline(train_X,
                                                                train_y_em,
                                                                len_img_train)
        report_grid_search(em_grid_search, em_pipe, em_params)
        best_em_clf = em_grid_search.best_estimator_

        print("E/M performance on the left out dataset: {0}".format(
            best_em_clf.score(test_X, test_y_em)))
        em_predictions = best_em_clf.predict(test_X)
        print(em_predictions)

        get_metrics(test_y_em, em_predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If no explicit training data, algorithm splits it itself
    parser.add_argument( "-d", "--data", help="pass a folder path to the data")

    # Otherwise
    parser.add_argument( "-t", "--train", help="pass a folder path to the training data")
    parser.add_argument( "-s", "--test", help="pass a folder path to the testing data")

    args = parser.parse_args()
    main(args)
