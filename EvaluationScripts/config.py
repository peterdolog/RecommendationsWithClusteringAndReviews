#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#configuration file for running evaluations in different options

seed = 0 #for which seed
folds = 1 #how many folds do we have. When 0, we only have one test and one train fold.
recommendationlistsize = 100 #size of the recommendation list considered for evaluation
#recommendationlistsize = 101 #size of the recommendation list considered for evaluation

ratingclass = 1 #which rating class we consider
whichratings = "allRatings" # implemented in dot prediction. Currently only allRatings or movieRatings and it is relevant for now only for SpectralMix. There are embeddings learning from only movie ratings and from all ratings
dimmensionality = 10
runningAllColumnOptions = 1 #with 0 runs for number of columns in dimmensionality, when 1 for all options up to dimmensionality variable
maxclusters = 5 #max number of clusters considered
minclusters = 2 #min number of clusters considered
runningAllClusterOptions = 0 #with 0 it just runs for number of clusters in clusters variable, with 0, all options up to clusters
#for evaluation, all which is needed in the table should be set to 1
#for other runs, only that option should be set to 1 which should be currently computed.
reviews = 1 #with 0, ignores review embeddings, with 1 - produces concatenation of review embeddings to items and users
clusters = 1 #with 0, ignores clustering embeddings, with 1 - produced dot embeddings with cluster centroids
concatclusters = 1 #with 0, ignores clustering embeddings, with 1 produces concatenation of cluster centroids to intitial embeddings
initialemb = 1 # with 1 considers initial embeddings, with 0 do not consider initial embeddings
withclustersandreviews = 1 #with 1 it considers reviews concatenated to the embeddings with clusters, with 0 not.
withreviewsandclusters = 1 #with 1 it considers clustering on embeddings concatenated with reviews, with 0 not.
spectralclustering = 1 # there is a specific behaviour in filtering users (joining.py). Set 1 when running spectral clustering for this
KGCL = 1 # there is a specific behaviour in filtering users (joining.py). Set 1 when running KGCL for this. The same is valid for LightGCN and BPR-MF, AdaCGL and CGCL
SpectralMix = 0 # there is a specific behaviour in filtering users (joining.py). Set 1 when running KGCL for this
#BaseFolder = "Data/eval/5FoldEval/BPR-MF/" #root folder for evaluation BPR-MF
#BaseFolder = "Data/eval/5FoldEval/LightGCN/" #root folder for evaluation LightGCN
#BaseFolder = "Data/eval/5FoldEval/KGCL/" #root folder for evaluation KGCL
#BaseFolder = "Data/eval/5FoldEval/KGCL_amazon/" #root folder for evaluation KGCL on amazon data
#BaseFolder = "Data/eval/5FoldEval/KGCL_amazon_101/" #root folder for evaluation KGCL on amazon data for lists of size 101 items to see whether it changes the results
#BaseFolder = "Data/eval/5FoldEval/KGCLamazonCGCLsplit/" #root folder for evaluation KGCL on amazon data  with split from CGCL traina nd test for lists of size 101 items to see whether it changes the results
#BaseFolder = "Data/eval/5FoldEval/KGCL_yelp/" #root folder for evaluation KGCL on amazon data
#BaseFolder = "Data/eval/5FoldEval/LightGCN_amazon/" #root folder for evaluation LightGCN on amazon data
#BaseFolder = "Data/eval/5FoldEval/LightGCN_amazon_KGCLdata/" #root folder for evaluation LightGCN on amazon data
#BaseFolder = "Data/eval/5FoldEval/LightGCN_yelp/" #root folder for evaluation LightGCN on amazon data


#BaseFolder = "Data/eval/5FoldEval/AdaGCL_Mindreader/" #root folder for evaluation AdaCGL on mindreader

#BaseFolder = "Data/eval/5FoldEval/AdaGCL_Amazon/" #root folder for evaluation AdaCGL on amazon

#BaseFolder = "Data/eval/5FoldEval/AdaCGL_yelp/" #root folder for evaluation AdaCGL on yelp


#BaseFolder = "Data/eval/5FoldEval/CGCL_Mindreader/" #root folder for evaluation CGCL on mindreader>>>>

#BaseFolder = "Data/eval/5FoldEval/CGCL_Amazon/" #root folder for evaluation CGCL on amazon

#BaseFolder = "Data/eval/5FoldEval/CGCL_yelp/" #root folder for evaluation CGCL on yelp


#BaseFolder = "Data/eval/5FoldEval/BPRMF_amazon/" #root folder for evaluation BPRMF on amazon
#BaseFolder = "Data/eval/5FoldEval/BPRMF_yelp/" #root folder for evaluation BPRMF on yelp

#BaseFolder = "Data/eval/5FoldEval/BPRMF_amazon_KGCLdata/" #root folder for evaluation BPRMF on amazon KGCL data

#BaseFolder = "Data/eval/5FoldEval/SpectralClustering_Amazon/" #root folder for evaluation spectral clustering
# BaseFolder = "Data/eval/5FoldEval/SpectralClustering_yelp/" #root folder for evaluation spectral clustering

#BaseFolder = "Data/eval/5FoldEval/" #root folder for evaluation without specific method folder


#BaseFolder = "Data/eval/5FoldEval/KGCL_yelp/KNNYelp/" #root folder for making KNN prediction baseline


#BaseFolder = "Data/eval/5FoldEval/SpectralMix/" #root folder for evaluation of Spectral Mix

#BaseFolder = "Data/eval/5FoldEval/SpectralMixCosineTimesEuclidean/" #root folder for evaluation of Spectral Mix Cosine times Euclidean

#BaseFolder = "Data/eval/5FoldEval/SpectralMixCosine/" #root folder for evaluation of Spectral Mix Cosine
#BaseFolder = "Data/eval/5FoldEval/ReviewsOnly/" #root folder for evaluation of reviews connected to the original dataset (for example movies or amazon books)
#BaseFolder = "Data/eval/5FoldEval/ReviewsOnly_amazon/" #root folder for evaluation of revies connected to the original dataset (for example movies or amazon books)
BaseFolder = "Data/eval/5FoldEval/ReviewsOnly_yelp/" #root folder for evaluation of reviews connected to the original dataset (for example movies or amazon books)
ReviewsOnly = 1 # 1 when only reviews are considered as embeddings, 0 otherwise
#SpectralMixEmbeddingsDir = BaseFolder + "SpectralMixTrainingEmbeddings/" #where the initial embeddings from SpectralMix should be located
SpectralMixEmbeddingsDir = BaseFolder + "InitialEmbeddings/" #where the initial embeddings from SpectralMix should be located - for spectral clustering it should be here - initial embeddings
SplitToClassDir = BaseFolder + "EmbeddingsSplitToRatingClasses/" #directory where a split of embeddings to different classes is saved
#SplitToClassDir = BaseFolder + "InitialEmbeddings/" #This is equal to embeddings dir since we are not filtering for different ratings in other methods than Spectral Mix
FoldsDir = BaseFolder + "folds/" #Where we have ground truth and train folds
PredictionResultsDir = BaseFolder + "PredictionResults/" #directory where we have prediction results after preprocesisngs (split to classes, clustering, and review embeddings fusions)
EvaluationResultsDir = BaseFolder + "EvaluationResults/" #directory for evaluation results with decided ratings
SelectedColumnsDir = BaseFolder + "EmbeddingsWithSelectedColumns/" #directory for embeddings only with selected columns
WithReviwesEmbeddingsDir = BaseFolder + "EmbeddignsWithReviews/" #directory for embeddings with reviews concatenated
# SentenceEmbDir = BaseFolder + "sentence_embeddings/text_embeddings28062022/sentence_embeddings_280622/" #for mindreader directory where the sentence embeddings for reviews are
SentenceEmbDir = BaseFolder + "sentence_embeddings/" # for amazon
SpectralClusteringEmbeddingDir = BaseFolder + "SpectralClusteringEmbedding/"
EmbeddingsFolderWithClusters = BaseFolder + "EmbeddingsWithClusters/" #folder for embeddings with clusters on original embeddings (split to classes)
SentenceEmbFile = "test_embeddings.pbz2" #file with sentence embeddings
jsonURIfile = "movie_uris.json" #file with mapping from URI to moviid
#GraphDataDir = BaseFolder + "SpectralClustering/graphfiles/"
GraphDataDir = BaseFolder + "graphfiles/recbolsplit/"
#GraphDataDir = BaseFolder + "graphfiles/"
