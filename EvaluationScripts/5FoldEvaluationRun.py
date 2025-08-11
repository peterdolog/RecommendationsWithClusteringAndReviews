
#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#1./ split embeddings to classes (joining.py - it also filters only those which are in the ground truth)
#1.a/ Remove columns from embeddings to find out how many features are needed by running dropcolumnsfromembeddings.py
#1.b/ Run Clustering to enhance embeddings of users and items (ClusteringEmbeddingsUsers.py for users and ClusteringEmbeddingsUsersItems.py)
#1.c/ Run concatenations of review embeddings to enhace embeddings of users by NLP produced review embeddings (concatreviewembeddings.py)
#2./ Perform prediction for a class (presumably positive) -
#SpectralMixRecommendationPrediction.py (dot product) SpectralMixRecommendationEuclidean.py (euclidean)
#3./ Run the evaluation: SpectralMixRecommendationEvaluation.py

import config
import concatreviewembeddings
import ClusteringEmbeddingsUsers
import ClusteringEmbeddingsUsersItems
import SpectralMixRecommendationPrediction
import SpectralMixRecommendationEuclidean
import SpectralMixRecommendationEvaluation
import datetime as dt



startdate = dt.datetime.now()
print("Starting the whole evaluation for Spectral Clustering: " + str(startdate))

#prediction needs to be done for all options separately
#clustering has 2 options: withclusterandreviews and with reviews and clusters
#concat has 2 options: with reviews and withclustersandreviews
#baselines folder need to be copied to the prediction folder

# initial embeddings-> concat reviews with reviews option -> clustering with withclustesandreviews
# -> clustering withreviewsandclusters -> concatreviews withclustersandreviews -> prediction with all options
# -> evaluation

#runnig prediction on initial embeddings
# 1/i.e
#config
config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 1 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 0 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.


#prediction
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()




#getting clustering over initial embeddings and prediction after
# 2/ clust(i.e.)
#config


config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 1 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.


#clustering
ClusteringEmbeddingsUsers.run()
ClusteringEmbeddingsUsersItems.run()



config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 1 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 0 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.

#runnig prediction after for clusters
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()



#running prediction for concat clusters
config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 1 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 0 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.

#runnig prediction after
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()



#config
config.reviews = 1 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 0 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.



#concatenating

# run on mindreader
#concatreviewembeddings.run()

#run on amazon or yelp
concatreviewembeddings.run_on_amazon()

#prediction
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()

#concatenating reviews and clusterings over users and items and prediction after
# 4/ clust(i.e.) + rev
#config
config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 1 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 0 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.


#concatenation
# run on mindreader
#concatreviewembeddings.run()

#run on amazon and yelp
concatreviewembeddings.run_on_amazon()

#prediction
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()



# clustering over initial embeddings with reviews
# 5/ clust(i.e.+rev)
#config

config.reviews = 0 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 0 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 0 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 0 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 0 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 1 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.



#clustering
ClusteringEmbeddingsUsers.run()
ClusteringEmbeddingsUsersItems.run()
#runnig prediction after
SpectralMixRecommendationPrediction.run()
SpectralMixRecommendationEuclidean.run()



config.reviews = 1 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
config.clusters = 1 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
config.concatclusters = 1 #eith 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
config.initialemb = 1 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
config.withclustersandreviews = 1 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
config.withreviewsandclusters = 1 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.

#running evaluation
SpectralMixRecommendationEvaluation.run()

enddate = dt.datetime.now()
print("Finished the whole evaluation for Spectral Mix: " + str(enddate))
print("Duration: " + str(enddate - startdate))
