# Recommendations With Clustering And Reviews
This is the github repository for the paper:

    <I>Peter Dolog, Sergio David Rico Torres, Yllka Velaj, Ylli Sadikaj, Andreas Stephan, Benjamin Roth, and Claudia
    Plant. 2025. The Impact of Graph Structure, Cluster Centroid and Text Review Embeddings on Recommendation
    Methods.</I>


We have used python 3.8 and associated libraries.

The main evaluation script is in the EvaluationScripts directory. The name is 5ForldEvaluationRun
It is used with 5 folds for Mindreader, but only 1 fold for Amazon and Yelp datasets since there we follow the same procedure as KGCL 2022 paper and also their split.

We learn embeddings with referenced methods and export them in the format with userid/itemid as a first column. This is expected as a format in our evaluation. Of course you can change it, but then you need to adapt the code.
The script the augments the embeddings with clusters and reviews as described in the paper
One run of script is for one method embeddings only.

The script expects directories as specified in the config file. This can be changes, but you then need to adapt the config file and maybe also code.

There are utils for aggregating from evaluations of each method after single method evaluation is done and cvs file is saved.

There are various config options which we suggest you review before starting.

There are also some assumptions on review embeddings. We have produced review embeddings per each sinlge review. Then we used util to aggregate it for users and items with userid/itemid as a first column for amazon and yelp.
The review embeddings for mindreader was created also per each review in the movie reviews but it required remapping to mindreader dataset urls, and only aggregate afterwards to remap to userids and itemids. That is why the aggregation is done in runtime of evaluation.

Before we run the recommendation prediction, we preselect from predictions only items and users from ground truth. There is a part which only related to mindreader and SpectralMix, since there we have also negative and neutral ratings. The utility for doing so is split_and_filter.
