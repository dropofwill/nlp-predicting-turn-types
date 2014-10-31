"""
Team Challenge: Predicting Turn Types
Authors: Tong, Will, and Ryan
"""
import os
import sys
import csv
import argparse
import pprint

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

pp = pprint.PrettyPrinter(indent=2)

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
                    # TODO: need to do some preprocessing here
                    # remove entire repair turns <>
                    # remove brackets around words []
                    X.append(row[5])
    return X, y_qa, y_em

def read_dir_dict(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    Returns an array of dicts
    """
    output = []
    X = []
    y_qa = []
    y_em = []
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

def main(args):
    # Our features and two sets of labels
    corpus, y_qa, y_em = read_dir_sk(args.data)

    y_qa, le_qa = encode_labels(y_qa)
    y_em, le_em = encode_labels(y_em)

    pp.pprint(corpus)

    #pp.pprint(le_qa.inverse_transform(y_qa))

    #pp.pprint(X)
    #pp.pprint(y_qa)
    #pp.pprint(y_em)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-d", "--data", help="pass a folder path to the data")
    args = parser.parse_args()
    main(args)
