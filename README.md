NLP Problem Set 3
===============

## TODO

More feature extraction techniques (see feature ideas)

Combining feature different feature selection techniques with a feature union, possibly in the pipeline, [like here](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html)?

Try feature selection to see how something like SelectKBest or the like would effect results (add it to the GridSearch?)

Try lots of different ML algorithms and their various tuning parameters available in Sklearn.

Split data further leaving one out to simulate the test environment.

Produce the necessary performance measures for reporting purposes.

Output CSV with our predictions in the same format as the input.

## Feature ideas:

Question words as features: how, what, where, when, who, etc.?

*Features that won't be image content specific:*

Shallow Semantic parsing

Name Entity Recognition

Syntactic features

## Reporting Deliverables

1. Title (including name/acronym of your team’s system implementation and the names of contributing team members on the first slide)

**System Name/Acronym?**

Ryan Dennehy, Tong Liu, & Will Paul


2. Overview of the team’s implementation approach (What you did); including both a **visualization** overview of your implementation and a **written description** of it

3. Results on predicting each of the target labels for the test data. Include these performance measures for your approach for each task:

    a. Table with accuracy on test data per task & per image

    b. Table with % error reduction over the majority class baseline on test data per task & per image

    c. Four confusion matrices (two per image with one per task)

4. Discussion of results with interpretation

5. Highlights (particularly interesting observations—this could involve further data analytics, successful specifics your implementation, additional interesting results, ‘case study’ of particular instances, etc.)

6. Challenges and how the team solved them

7. Task distribution (who did what, and estimated % effort per person)

8. Conclusion

9. References, as applicable
>>>>>>> master
