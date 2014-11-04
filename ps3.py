"""
Team Challenge: Predicting Turn Types
Authors: Tong, Will, and Ryan

Expects a path to a directory of CSVs, either one directory for K-fold cross
validation or two folders, one for training and one for testing.

    --data  Path to single directory for tuning
or
    --train To directory for training data
    --test  To directory for testing data

When a test directory is given it produces duplicates of those files, except
this time with it's own predictions for the Q/A and E/M tasks.

Otherwise it prints to the terminal information about its CV performance.
"""

import os, re, sys, csv, argparse, logging, nltk
from pprint import pprint
from time import time

from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

from sklearn import base
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, f_classif

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm, metrics

from nltk import word_tokenize

# Hardcode global length of an image document
len_img = 40

def read_csv(path):
    output = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            output.append(row)
    return output

def write_csv(path, subjectID, imageID, questionID, qa, em, text):
    with open(path, "wb") as f:
        writer = csv.writer(f, delimiter=",", dialect="excel")
        for i in range(len(subjectID)):
            writer.writerow([subjectID[i], imageID[i], questionID[i],
                            qa[i], em[i], text[i]])
    return text

def export_submission_csv(  filenames, subjectID, imageID, questionID,
                            qa, em, text, breakpoint=len_img):
    s_i = 0
    for i, f in enumerate(filenames):
        e_i = (i+1) * breakpoint
        write_csv(f, subjectID[s_i:e_i], imageID[s_i:e_i],
                     questionID[s_i:e_i], qa[s_i:e_i],
                     em[s_i:e_i], text[s_i:e_i])
        #print("length check")
        #print(s_i, e_i)
        s_i = e_i

def read_dir_sk(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    Returns an array of dicts, array of qa labels, and an array of em labels
    """
    X, X_pos, y_qa, y_em, sID, iID, qID, fID = [], [], [], [], [], [], [], []

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
                    token = text.split()
                    pos_tag = nltk.pos_tag(token)
                    X_pos.append(pos_tag)
                    sID.append(row[0])
                    iID.append(row[1])
                    qID.append(row[2])
    return sID, iID, qID, X, X_pos, y_qa, y_em, fID

class POSTransformer():
    """
    A stateless transformer that wraps the pos_features method
    tokens_to_replace takes a list of string representation of tags, which are
        used instead of the token for ngrams
    """

    def __init__(self,
                tokens_to_replace=["NN"],
                no_replace=[]):
        self.tokens_to_replace = tokens_to_replace
        self.no_replace = no_replace

    def transform(self, X):
        return self.pos_features(X)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {"tokens_to_replace": self.tokens_to_replace,
                "no_replace": self.no_replace}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def pos_features(self, pos_tag):
        newtext, Xt = [], []

        for utterance in pos_tag:
            newtext = []

            for item in utterance:
                word = item[0]
                pos = item[1]

                for tag in self.tokens_to_replace:
                    if pos == tag:
                        if word not in self.no_replace:
                            newtext.append(pos)
                        else:
                            newtext.append(word)
                    else:
                        newtext.append(word)
            Xt.append(newtext)
        return Xt

class TrigramPOSTransformer():
    """
    Create all possible trigram POS/Token instantiation levels (2^3 = 8 states)
    Trigram possible states:
    0 0 0
    0 1 0
    0 0 1
    0 1 1
    1 0 0
    1 1 0
    1 0 1
    1 1 1
    """
    def __init__(self):
        self.possible_states = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1]
        ]

    def transform(self, X):
        return self.pos_features(X)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def _trigrams(self, tokens):
        """
        Turn tokens into a list of trigrams
        """
        min_n, max_n = 3, 3
        if max_n != 1:
            original_tokens = tokens
            tokens = []
            n_original_tokens = len(original_tokens)
            for n in xrange(min_n,
                            min(max_n + 1, n_original_tokens + 1)):
                for i in xrange(n_original_tokens - n + 1):
                    tokens.append(original_tokens[i: i + n])
        return tokens

    def _filter_trigrams(self, trigrams):
        """
        Retrieve every possible combination of token/pos trigrams (8 each)
        """
        new_trigrams = []

        for trigram in trigrams:
            for state in self.possible_states:
                new_tri = []

                for i, state in enumerate(state):
                    new_tri.append(trigram[i][state])

                new_trigrams.append(" ".join(new_tri))
        return new_trigrams

    def pos_features(self, pos_tag):
        newtext, Xt = [], []

        for utterance in pos_tag:
            trigrams = self._trigrams(utterance)
            trigrams = self._filter_trigrams(trigrams)
            Xt.append(trigrams)
        return Xt

class QATransformer():
    """
    A stateless transformer that wraps the q_features method
    """
    def __init__(self, begin=0, end=2):
        self.begin = begin
        self.end = end

    def transform(self, X):
        return self.q_features(X, self.begin, self.end)

    def fit(self, X, y=None):
        return self

    def get_params(self, deep=True):
        return {"begin": self.begin, "end": self.end}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

    def q_features(self, X, begin=0, end=2):
        """
        Takes a list of raw strings as input
        Returns a dict of binary features that imply the sentence is a question
        """
        wh_words = [    # Wh-pronouns
                        "what", "which", "who", "whom",
                        # Wh-possesive
                        "whose",
                        # Wh-adverb
                        "how", "when", "where", "why",
                        # Wh-* with 's
                        "what's", "who's"]

        ans_words = [   # non-speech disfluencies
                        "sp", "{sl}", "{ls}", "{cg}", "{ns}", "{br}",
                        # speech disfluencies
                        "uh", "um", "hm", "mm"
                        # conjunctions and sentence starter words
                        "well", "and", "but", "yet"]

        aux_verbs = ["am", "is", "are", "was", "were",
                    "have", "had", "has",
                    "do", "does", "did"]

        modal_verbs = ["can", "could",
                      "may", "might", "must",
                      "shall", "should",
                      "will", "would"]
        features = []
        text = []
        for sentence in X:
            per_utterance = dict()
            tokens = sentence.split()
            text.append(tokens)

            for i in range(begin, end + 1):
                if len(tokens) > i + 1:
                    for wh in wh_words:
                        if tokens[i] == wh:
                            per_utterance[wh+"_"+str(i)] = 1

                    for aux in aux_verbs:
                        if tokens[i] == aux:
                            # Only match if the first, non ans_word
                            if i != begin:
                                ok = True
                                for j in range(begin, i):
                                    if tokens[j] not in ans_words:
                                        ok = False
                                if ok:
                                    per_utterance[aux+"_"+str(i)] = 1
                            else:
                                per_utterance[aux+"_"+str(i)] = 1

                    for modal in modal_verbs:
                        if tokens[i] == modal:
                            # Only match if the first, non ans_word
                            if i != begin:
                                ok = True
                                for j in range(begin, i):
                                    if tokens[j] not in ans_words:
                                        ok = False
                                if ok:
                                    per_utterance[modal+"_"+str(i)] = 1
                            else:
                                per_utterance[modal+"_"+str(i)] = 1
            features.append(per_utterance)
        return features


def POS_feature_convertor(pos_tag, tokens_to_replace=['VB']):
    newtext = []
    match = False

    # token = text.split()
    # pos_tag = nltk.pos_tag(token)

    for item in pos_tag:
        word = item[0]
        pos = item[1]

        for tag in tokens_to_replace:
            if pos == tag:
                newtext.append(pos)
                match = True
            else:
                newtext.append(word)
    return newtext

def encode_labels(y, le=None):
    """
    Takes a list of labels and converts them to numpy compatible classes
    Returns (the converted labels, and the encoder)
    """
    if le:
        y = le.fit_params(y)
    else:
        le = LabelEncoder()
        y = le.fit_transform(y)
    return y, le

def decode_labels(y, le):
    return le.inverse_transform(y)

def grid_search_pipeline(pipe, params, cv, data, targets):
    """
    For a given pipeline, parameters, cross validator, dataset, and target labels
    Run a grid search using the given pipeline and cross validator over the set
    of parameters, printing the progress as it goes.
    Returns the results
    """

    grid_search = GridSearchCV(pipe, params, cv=cv, verbose=1, n_jobs=3)

    print("Performing grid search...")
    #print("pipe:", [name for name, _ in pipe.steps])
    #print("params:")
    #pprint(params)
    print
    t0 = time()
    grid_search.fit(data, targets)
    print("done in %0.3fs" % (time() - t0))

    return grid_search

def POS_svm_pipeline(data, targets, num_images=11):
    """
    A simple pipeline using POS as features, support vector machine, and KFold
    cross validation such that on each run one image file is left out (set
    num_images differently if working on a smaller training set)
    Returns the grid_search results, pipeline, and parameters used
    """

    pipe = Pipeline([
        ("features", FeatureUnion([
            ("nn_pipe", Pipeline([
                ("nn_preprocess", POSTransformer(
                    tokens_to_replace=["NN", "NNS", "NNP", "NNPS"],)),
                ("nn_vect", TfidfVectorizer(   preprocessor=pass_input,
                                               tokenizer=pass_input,
                                               # From previous CV runs:
                                               use_idf=True,
                                               sublinear_tf=True,
                                               smooth_idf=True))
            ])),

            ("cc_pipe", Pipeline([
                ("cc_preprocess", POSTransformer(
                    tokens_to_replace=["CD", "CC", "LS"])),
                ("cc_vect", TfidfVectorizer(   preprocessor=pass_input,
                                               tokenizer=pass_input,
                                               use_idf=True,
                                               sublinear_tf=True,
                                               smooth_idf=True))
            ])),

            ("jj_pipe", Pipeline([
                ("jj_preprocess", POSTransformer(
                    tokens_to_replace=["JJ", "JJR", "JJS"])),
                ("jj_vect", TfidfVectorizer(   preprocessor=pass_input,
                                               tokenizer=pass_input,
                                               use_idf=True,
                                               sublinear_tf=True,
                                               smooth_idf=True))
            ])),
        ])),
        ("selection", SelectKBest(f_classif, k=2000)),
        #("clf", svm.LinearSVC())
        ("clf", svm.SVC())
        #("clf", MultinomialNB())
    ])

    pipe = Pipeline([
        ("preprocess", POSTransformer()),
        ("vect", TfidfVectorizer(   preprocessor=pass_input,
                                    tokenizer=pass_input,
                                    # From previous CV runs:
                                    use_idf=True,
                                    sublinear_tf=True,
                                    smooth_idf=True)),
        #("selection", SelectKBest(f_classif, k=1000)),
        # Performed better than SelectKBest in the 25-75 percentile range
        ("selection", SelectPercentile(f_classif)),
        ("clf", svm.LinearSVC())
        #("clf", svm.SVC())
        #("clf", MultinomialNB())
    ])

    pipe = Pipeline([
        ("preprocess", TrigramPOSTransformer()),
        #("vect", CountVectorizer(   preprocessor=pass_input,
                                    #tokenizer=pass_input,
                                    #ngram_range=(3,3))),
        ("vect", CountVectorizer(analyzer=pass_input)),
        #("selection", SelectKBest(f_classif, k=1000)),
        # Performed better than SelectKBest in the 25-75 percentile range
        ("selection", SelectPercentile(f_classif)),
        ("clf", svm.LinearSVC(loss="l2"))
        #("clf", svm.SVC())
        #("clf", MultinomialNB())
    ])


    params = {
        #"preprocess__tokens_to_replace": (
            #["NN", "NNS"],
            #["NNP", "NNPS"],
            #["JJ", "JJR", "JJS"],
            #["RB", "RBR", "RBS"],
            #["CD", "LS"],
            #["PRP", "PRP$"],
            #["CC"],
            #["CD", "CC", "LS", "NNP", "NNPS", "NN", "NNS", "JJ", "JJR", "JJS"],
            #["CD", "CC", "LS"],
            #["NN", "NNS", "NNP", "NNPS"]
            #["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
            # All tags:
            #["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"]
        #),
        #"preprocess__no_replace": (["sp", "{sl}", "{ls}", "{cg}", "{ns}", "{br}", "uh", "um", "hm", "mm"], []),
        # Small effect, slightly better w/tfidf
        #"vect__use_idf": (True, False),
        # Small effect, slightly better w/sublinear
        #"vect__sublinear_tf": (True, False),
        # Small effect, slightly better Std w/smoothing
        #"vect__smooth_idf": (True, False),
        #"vect__ngram_range": (
            #(1, 1),
            #(1, 2),
            #(1, 3),
            #(2, 2),
            #(2, 3),
            #(3, 3),
            #(2, 4),
            #(3, 4),
            #(4, 4)
        #),
        # "vect__norm": ("l1", "l2")
        # "vect__stop_words": ("english", None)
        #"selection__k": (1500, 2000, "all")
        "selection__percentile": (25, 50, 75),
        "clf__C": (0.1, 1.0, 10.0),
        #"clf__loss": ("l1", "l2"),
        #"clf__penalty": ("l1", "l2"),
        #"clf__dual": (True, False),
        "clf__tol": (1, 1e-1, 1e-2, 1e-3, 1e-4),
        #"clf__fit_intercept": (True, False),
        #"clf__kernel": ("linear", "poly", "rbf")
    }

    cv = KFold(len(targets), num_images)
    grid_search = grid_search_pipeline(pipe, params, cv, data, targets)
    return grid_search, pipe, params


def qa_mnb_pipeline(data, targets, num_images=11):
    """
    A Q/A specific pipeline
    """
    # Combine tfidf ngram and qa dict features and pass to a single clf
    pipe = Pipeline([
        ("features", FeatureUnion([
            ("qa_pipe", Pipeline([
                ("qa_trans", QATransformer()),
                ("dict_vect", DictVectorizer()),
            ])),
            ("tfidf", TfidfVectorizer(stop_words="english"))
        ],
        # Weight the syntax rules more heavily then the ngrams
        transformer_weights={"qa_pipe": 7, "tfidf": 3}
        )),
        ("selection", SelectKBest()),
        ("clf", MultinomialNB())
    ])

    params = {
        "clf__alpha": (1, 0.1, 0.001, 0.00001, 0.000001),
        # Claims that FeatureUnion has no attr transformer weights
        #"features__transformer_weights": (  None,
                                            #{"qa_pipe": 0.25, "tfidf": 0.75},
                                            #{"qa_pipe": 0.75, "tfidf": 0.25}),
        # Chain pipeline methods with double underscores
        "features__qa_pipe__qa_trans__end": (1, 2, 3),
        "selection__k": (10, 100, 500, "all")
    }

    cv = KFold(len(targets), num_images)
    grid_search = grid_search_pipeline(pipe, params, cv, data, targets)
    return grid_search, pipe, params

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
        "vect__max_df": (0.5, 0.75, 1.0),
        # "vect__max_features": (None, 5000, 10000, 50000),
        "vect__use_idf": (True, False),
        "vect__analyzer": (
            "word",
            "char_wb"
            #"char"
            ),
        "vect__ngram_range": (
            (1, 1),
            (1, 2),
            (2, 2),
            (1, 3),
            (2, 3),
            (3, 3)
        ),
        "vect__norm": ("l1", "l2"),
        "clf__alpha": (1, 0.1, 0.001, 0.00001, 0.000001)
    }

    cv = KFold(len(targets), num_images)
    grid_search = grid_search_pipeline(pipe, params, cv, data, targets)
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


def report_grid_scores(grid_scores, n_top=20):
    """
    Helper function to report score performance of the top n classifier / params
    """
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("{2}. Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores),
            i + 1))
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

def pass_input(input):
    """
    Needed to override default preprocessor in Skleran *Vectorizers
    """
    return input

def main(args):
    if (args.data):
        print("Performing pos tagging...")
        # Our features and two sets of labels
        s1, i1, q1,             \
        train_X, train_X_pos,   \
        train_y_qa, train_y_em, _ = read_dir_sk(args.data)

        len_img_train = int(float(len(train_y_qa))/float(len_img))
        train_y_qa, le_qa = encode_labels(train_y_qa)
        train_y_em, le_em = encode_labels(train_y_em)

        #print POS_feature_convertor(train_X_pos[0])
        # pos_vectorizer = TfidfVectorizer(preprocessor=pass_input, tokenizer=POS_feature_convertor, ngram_range=(1, 2))
        # print pos_vectorizer.fit_transform(train_X_pos).toarray().shape

        print("--- Q/A ---")
        # Tong's SVM Token/POS pipeline
        #qa_grid_search, qa_pipe, qa_params = POS_svm_pipeline(train_X_pos, train_y_qa)
        #report_grid_search(qa_grid_search, qa_pipe, qa_params)
        #best_qa_clf = qa_grid_search.best_estimator_

        # Will's MNB Syntax Rules > Ngrams pipeline
        #qa_grid_search, qa_pipe, qa_params = qa_mnb_pipeline(train_X,
                                                            #train_y_qa,
                                                            #len_img_train)
        #report_grid_search(qa_grid_search, qa_pipe, qa_params)

        # Basic TFIDF feature set of Ngrams
        #qa_grid_search, qa_pipe, qa_params = tfidf_mnb_pipeline(train_X, train_y_qa)
        #report_grid_search(qa_grid_search, qa_pipe, qa_params)
        #best_qa_clf = qa_grid_search.best_estimator_

        print
        print("--- E/M ---")
        #print(train_X_pos)
        # Tong's SVM Token/POS pipeline
        em_grid_search, em_pipe, em_params = POS_svm_pipeline(train_X_pos, train_y_em)
        report_grid_search(em_grid_search, em_pipe, em_params)
        best_em_clf = em_grid_search.best_estimator_

        # Basic TFIDF feature set of Ngrams
        #em_grid_search, em_pipe, em_params = tfidf_mnb_pipeline(train_X, train_y_em)
        #report_grid_search(em_grid_search, em_pipe, em_params)
        #best_em_clf = em_grid_search.best_estimator_

    elif (args.test and args.train):
        print("Performing pos tagging...")

        s1, i1, q1,             \
        train_X, train_X_pos,   \
        train_y_qa, train_y_em, _ = read_dir_sk(args.train)

        s2, i2, q2,             \
        test_X, test_X_pos,     \
        test_y_qa, test_y_em, test_f_names = read_dir_sk(args.test)

        train_y_qa, le_qa = encode_labels(train_y_qa)
        train_y_em, le_em = encode_labels(train_y_em)
        test_y_qa, _ = encode_labels(test_y_qa)
        test_y_em, _ = encode_labels(test_y_em)

        train_y_qa, _ = encode_labels(train_y_qa)
        train_y_em, _ = encode_labels(train_y_em)
        test_y_qa, _ = encode_labels(test_y_qa)
        test_y_em, _ = encode_labels(test_y_em)

        em_baseline_prob = find_baseline(train_y_em)
        qa_baseline_prob = find_baseline(train_y_qa)

        # how many documents are in the training set?
        len_img_train = int(float(len(train_y_qa))/float(len_img))
        len_img_test = int(float(len(test_y_qa))/float(len_img))
        #print(len_img_train, len_img_test)

        print
        print("--- Q/A ---")
        qa_grid_search, qa_pipe, qa_params = qa_mnb_pipeline(train_X,
                                                            train_y_qa,
                                                            len_img_train)
        report_grid_search(qa_grid_search, qa_pipe, qa_params)
        best_qa_clf = qa_grid_search.best_estimator_

        print
        print("Q/A baseline {0}".format(qa_baseline_prob))
        print("Q/A performance on the left out dataset: {0}".format(
                best_qa_clf.score(test_X, test_y_qa)))

        qa_predictions = best_qa_clf.predict(test_X)
        qa_prob_predictions = best_qa_clf.predict_proba(test_X)
        #print("QA prediction arrays:")
        #print(qa_predictions)
        #print(qa_prob_predictions)

        print
        print("Q/A Metrics for Image 1")
        get_metrics(qa_baseline_prob, test_y_qa[:40], qa_predictions[:40], False)

        print
        print("Q/A Metrics for Image 2")
        get_metrics(qa_baseline_prob, test_y_qa[40:], qa_predictions[40:], False)

        print
        print("--- E/M ---")
        em_grid_search, em_pipe, em_params = POS_svm_pipeline(  train_X_pos,
                                                                train_y_em,
                                                                len_img_train)

        #em_grid_search, em_pipe, em_params = tfidf_mnb_pipeline(train_X,
                                                                #train_y_em,
                                                                #len_img_train)
        report_grid_search(em_grid_search, em_pipe, em_params)
        best_em_clf = em_grid_search.best_estimator_

        print
        print("E/M baseline {0}".format(em_baseline_prob))
        print("E/M performance on the left out dataset: {0}".format(
            best_em_clf.score(test_X_pos, test_y_em)))

        em_predictions = best_em_clf.predict(test_X_pos)
        #em_prob_predictions = best_em_clf.predict_proba(test_X_pos)
        #print("EM prediction arrays:")
        #print(em_predictions)
        #print(em_prob_predictions)

        print
        print("E/M Metrics for Image 1")
        get_metrics(em_baseline_prob, test_y_em[:40], em_predictions[:40], False)

        print
        print("E/M Metrics for Image 2")
        get_metrics(em_baseline_prob, test_y_em[40:], em_predictions[40:], False)

        qa_human = decode_labels(qa_predictions, le_qa)
        em_human = decode_labels(em_predictions, le_em)

        export_submission_csv(  test_f_names, s2, i2, q2,
                                qa_human, em_human, test_X)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # If no explicit training data, algorithm splits it itself
    parser.add_argument("-d", "--data", help="pass a folder path to the data")

    # Otherwise
    parser.add_argument("-t", "--train", help="pass a folder path to the training data")
    parser.add_argument("-s", "--test", help="pass a folder path to the testing data")

    args = parser.parse_args()
    main(args)
