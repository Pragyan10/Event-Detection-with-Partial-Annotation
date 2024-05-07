# The readme detail is extracted from the original paper github repository - the authors of the paper have full copyright and ownership of the code base and the details in the readme


Code for Event Detection with Partial Annotation

====

We develop a model that treats ED as a trigger localization problem (similar to a Machine Reading Comprehension model). The runing steps are:

Run mrc_preprocessing.py for preprocessing.
Run mrc_train.py for training with different ratios of labeled data.
Run mrc_evaluation.py for evaluation a particular model.
We also provide code for the model SeqBERT for training, which has a similar training steps.

partial_labeled_data.zip contains the unlabelled event instances.
