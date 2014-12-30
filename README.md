Predicting Turn Types
===============

Team Members:

* Ryan Dennehy

* Tong Liu

* Will Paul

## Usage

Expects a path to a directory of CSVs, either one directory for K-fold cross
validation or two folders, one for training and one for testing.

    --data  Path to single directory for tuning
or
    --train To directory for training data
    --test  To directory for testing data

When a test directory is given it produces duplicates of those files, except
this time with it's own predictions for the Q/A and E/M tasks.

Otherwise it prints to the terminal information about its CV performance.

## TODO

√ More feature extraction techniques (see feature ideas)

√ Combining feature different feature selection techniques with a feature union, possibly in the pipeline, [like here](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html)?

√ Try feature selection to see how something like SelectKBest or the like would effect results (add it to the GridSearch?)

√ Try lots of different ML algorithms and their various tuning parameters available in Sklearn.

## Feature ideas:

#### Ways to form questions in English:

Move the auxiliary verb to beginning of sentence, Subject-auxiliary inversion:

It/PRP is/VBZ snowing/VBG vs. is/VBZ it/PRP snowing?/VBG

Move a modal to the beginning of the sentence.

They/PRP will/MD come/VB vs. Will/MD they/PRP come/VBP

Adding a Wh-\* (WDT, WP, WP$, WRB, in treebank) to the beginning of a sentence, also involves some other syntax rules, Wh-fronting (by far the most common in our dataset)

she/PRP often/RB uses/NNS it/PRP  vs. how/WRB often/RB does/VBZ she/PRP use/NN it?/PRP

Wh-\* tag in within the first 3 tokens of the sentence

[am, is, are, was, were, have, had, has, do, does, did] within the first 3 tokens of the sentence, but not after a NN\* tag

[can, could, may, might, must, shall, should, will, would] within the first 3 tokens of the sentence, but not after a NN\* tag

### Features that won't be image content specific:

√ Syntactic features

Shallow Semantic parsing

Name Entity Recognition
