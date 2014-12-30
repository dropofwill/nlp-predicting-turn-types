Predicting Turn Types
===============

Team Members:

* Ryan Dennehy

* Tong Liu

* Will Paul

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
