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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import metrics

# Hardcode global length of an image document
len_img = 40

def read_csv(path):
    output = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            output.append(row)
    return output

def write_csv(path, subjectID, imageID, questionID, QA, EM, text):
    with open(path, "wb") as f:
        writer = csv.writer(f,delimiter=",",dialect="excel")
        for i in range(len(subjectID)):
            writer.writerow([subjectID[i], imageID[i], questionID[i], QA[i], EM[i], text[i]])
    return text

def read_dir_sk(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    Returns an array of dicts, array of qa labels, and an array of em labels
    """
    X = []
    y_qa = []
    y_em = []
    sID = []
    iID = []
    qID = []
    fID = []
    for root, subdirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                a_file_path = os.path.join(root, f)
                csv = read_csv(a_file_path)
                fID.append(f)

                for row in csv:
                    y_qa.append(row[3])
                    y_em.append(row[4])
                    # remove asterisk *
                    # remove entire repair turns <>
                    # remove brackets around words []
                    text = re.sub(r'(<|>|\[|\]|\*)', '', row[5])
                    X.append(text)
                    sID.append(row[0])
                    iID.append(row[1])
                    qID.append(row[2])
    return sID, iID, qID, X, y_qa, y_em, fID

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
        "vect__analyzer": ("word", "char"),
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
        print

def find_baseline(y_label):
    count_1 = 0.0
    count_2 = 0.0

    for item in y_label:
        if item == 0:
            count_1 += 1
        else:
            count_2 += 1

    if count_1 >= count_2:
        baseline_prob = count_1 / float(len(y_label))
    else:
        baseline_prob = count_2 / float(len(y_label))

    return baseline_prob

def get_metrics(baseline_prob, y_test_list, y_pred_list, plot_results=True):
    y_true = y_test_list

    y_pred = y_pred_list

    accuracy = accuracy_score(y_true, y_pred)
    print 'Accuracy on test data: ' + str(accuracy)

    reduction = accuracy - baseline_prob
    print 'Error reduction over the majority class baseline: ' + str(reduction)

    # report the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print 'Confusion Matrix:\n' + str(cm)

    # plot confusion matrix in color in a separate window
    if (plot_results):
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
    if (plot_results):
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
        s1, i1, q1, train_X, train_y_qa, train_y_em, _ = read_dir_sk(args.train)
        s2, i2, q2, test_X, test_y_qa, test_y_em, test_f_names = read_dir_sk(args.test)

        print(test_f_names)

        train_y_qa, le_qa = encode_labels(train_y_qa)
        train_y_em, le_em = encode_labels(train_y_em)
        test_y_qa, le_qa = encode_labels(test_y_qa)
        test_y_em, le_em = encode_labels(test_y_em)

        train_y_qa, le_qa = encode_labels(train_y_qa)
        train_y_em, le_em = encode_labels(train_y_em)
        test_y_qa, le_qa = encode_labels(test_y_qa)
        test_y_em, le_em = encode_labels(test_y_em)

        em_baseline_prob = find_baseline(train_y_em)
        qa_baseline_prob = find_baseline(train_y_qa)
        print("Q/A baseline {0}".format(qa_baseline_prob))
        print("E/M baseline {0}".format(em_baseline_prob))

        # how many documents are in the training set?
        len_img_train = int(float(len(train_y_qa))/float(len_img))
        len_img_test = int(float(len(test_y_qa))/float(len_img))
        print(len_img_train, len_img_test)

        print("--- Q/A ---")
        qa_grid_search, qa_pipe, qa_params = tfidf_mnb_pipeline(train_X,
                                                                train_y_qa,
                                                                len_img_train)
        report_grid_search(qa_grid_search, qa_pipe, qa_params)
        best_qa_clf = qa_grid_search.best_estimator_

        print("Q/A performance on the left out dataset: {0}".format(
            best_qa_clf.score(test_X, test_y_qa)))
        qa_predictions = best_qa_clf.predict(test_X)
        qa_prob_predictions = best_qa_clf.predict_proba(test_X)
        print("QA prediction arrays:")
        print(qa_predictions)
        print(qa_prob_predictions)
        print

        print("Metrics for Q/A task on Image 1")
        get_metrics(qa_baseline_prob, test_y_qa[:40], qa_predictions[:40], False)
        print("Metrics for Q/A task on Image 2")
        get_metrics(qa_baseline_prob, test_y_qa[40:], qa_predictions[40:], False)

        print("--- E/M ---")
        em_grid_search, em_pipe, em_params = tfidf_mnb_pipeline(train_X,
                                                                train_y_em,
                                                                len_img_train)
        report_grid_search(em_grid_search, em_pipe, em_params)
        best_em_clf = em_grid_search.best_estimator_

        print("E/M performance on the left out dataset: {0}".format(
            best_em_clf.score(test_X, test_y_em)))
        em_predictions = best_em_clf.predict(test_X)
        em_prob_predictions = best_em_clf.predict_proba(test_X)
        print("EM prediction arrays:")
        print(em_predictions)
        print(em_prob_predictions)
        print

        print("Metrics for E/M task on Image 1")
        get_metrics(em_baseline_prob, test_y_em[:40], em_predictions[:40], False)
        print("Metrics for E/M task on Image 2")
        get_metrics(em_baseline_prob, test_y_em[40:], em_predictions[40:], False)

        s_i = 0
        for i, f in enumerate(test_f_names):
            e_i = (i+1) * 40
            write_csv(f, s2[s_i:e_i],
                         i2[s_i:e_i],
                         q2[s_i:e_i],
                         qa_predictions[s_i:e_i],
                         em_predictions[s_i:e_i],
                         test_X[s_i:e_i])
            print("length check")
            print(s_i, e_i)
            s_i = e_i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If no explicit training data, algorithm splits it itself
    parser.add_argument( "-d", "--data", help="pass a folder path to the data")

    # Otherwise
    parser.add_argument( "-t", "--train", help="pass a folder path to the training data")
    parser.add_argument( "-s", "--test", help="pass a folder path to the testing data")

    args = parser.parse_args()
    main(args)
