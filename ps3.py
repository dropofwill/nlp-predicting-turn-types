"""
Team Challenge: Predicting Turn Types
Authors: Tong, Will, and Ryan
"""
import os
import sys
import csv
import argparse
import numpy as np
import pprint

pp = pprint.PrettyPrinter(indent=4)

def read_csv(path):
    output = []
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            output.append(row)
    return output

def read_dir(path):
    """
    Takes a path to a directory of csv data files, parses them individually,
    and returns an array of the results
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

def main(args):
    data = read_dir(args.data)

    #f = open(args.data)
    #data = np.loadtxt(f, dtype="{'")

    pp.pprint(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "-d", "--data", help="pass a folder path to the data")
    args = parser.parse_args()
    main(args)
