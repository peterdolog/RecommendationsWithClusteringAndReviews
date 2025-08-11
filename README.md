# Recommendations With Clustering And Reviews
This is the github repository for the paper:

Peter Dolog, Sergio David Rico Torres, Yllka Velaj, Ylli Sadikaj, Andreas Stephan, Benjamin Roth, and Claudia
Plant. 2025. The Impact of Graph Structure, Cluster Centroid and Text Review Embeddings on Recommendation
Methods.

The main evaluation script is in the EvaluationScripts directory. The name is 5ForldEvaluationRun
It is used with 5 folds for Mindreader, but only 1 fold for Amazon and Yelp datasets since there we follow the same procedure as KGCL 2022 paper and also their split.

We learn embeddings with referenced methods and export them in the format with userid/itemid as a first column. This is expected as a format in our evaluation. Of course you can change it, but then you need to adapt the code.
The script the augments the embeddings with clusters and reviews as described in the paper
One run of script is for one method embeddings only.

The script expects directories as specified in the config file. This can be changes, but you then need to adapt the config file and maybe also code.

There are utils for aggregating from evaluations of each method after single method evaluation is done and cvs file is saved.

There are various config options which we suggest you review before starting.
