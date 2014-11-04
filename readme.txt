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
