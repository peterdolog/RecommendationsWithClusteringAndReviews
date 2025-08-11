#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#the script runs top-k evaluation. Requires the recommendation file with user, item, score, rank
#it requires truth file with item and user.
#it produces aggregated measurement for top k results
#if k-fold is run, it requires to be run k-times, one per fold, and ten aggregate


import pandas as pd
from lenskit import batch, topn, util
import datetime as dt
import config as cfg
import matplotlib


def run():


    start = dt.datetime.now()

    print("Time of start: " + str(start))


    results_mean = pd.DataFrame()
    results_mean_agg = pd.DataFrame()


    if cfg.concatclusters:
        #aggregated per fold
        filepath_write_results_agg_perfold_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfolds_ratingclass" + str( cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "aggregatedperfold_with_clusters_and_reviews_and_concatclusters.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfold_ratingclass" + str(cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "allfoldsaggregated_with_clusters_and_reviews_and_concatclusters.csv"

    if cfg.withclustersandreviews:
        #dotclustes
        #aggregated per fold
        filepath_write_results_agg_perfold_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfolds_ratingclass" + str( cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "aggregatedperfold_with_clusters_and_reviews.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfold_ratingclass" + str(cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "allfoldsaggregated_with_clusters_and_reviews.csv"
        # concatclusters
        # aggregated per fold
        filepath_write_results_agg_perfold_with_clusters_and_reviews_concat = cfg.EvaluationResultsDir + "seed" \
                    + str(cfg.seed) + "/evalaggregated_seed" + str(cfg.seed) + "_allfolds_ratingclass" \
                    + str(cfg.ratingclass) + "_traningratings_" + str(cfg.whichratings) \
                                        + "aggregatedperfold_with_clusters_and_reviews_concatclusters.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_with_clusters_and_reviews_concat = cfg.EvaluationResultsDir + "seed" \
                    + str(cfg.seed) + "/evalaggregated_seed" + str(cfg.seed) + "_allfold_ratingclass" \
                    + str(cfg.ratingclass) + "_traningratings_" + str(cfg.whichratings) \
                    + "allfoldsaggregated_with_clusters_and_reviews_concatclusters.csv"


    if cfg.withreviewsandclusters:
        # dotclustes
        #aggregated per fold
        filepath_write_results_agg_perfold_with_reviews_and_clusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfolds_ratingclass" + str( cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "aggregatedperfold_with_clusters_and_reviews.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_with_reviews_and_clusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
            + "/evalaggregated_seed" + str(cfg.seed) + "_allfold_ratingclass" + str(cfg.ratingclass) \
            + "_traningratings_" + str(cfg.whichratings) + "allfoldsaggregated_with_clusters_and_reviews.csv"
        #concatclusters
        # aggregated per fold
        filepath_write_results_agg_perfold_with_reviews_and_clusters_concat = cfg.EvaluationResultsDir + "seed" \
                    + str(cfg.seed) + "/evalaggregated_seed" + str(cfg.seed) + "_allfolds_ratingclass" \
                    + str(cfg.ratingclass) + "_traningratings_" + str(cfg.whichratings) \
                                        + "aggregatedperfold_with_clusters_and_reviews_concatclusters.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_with_reviews_and_clusters_concat = cfg.EvaluationResultsDir + "seed" \
                    + str(cfg.seed) + "/evalaggregated_seed" + str(cfg.seed) + "_allfold_ratingclass" \
                    + str(cfg.ratingclass) + "_traningratings_" + str(cfg.whichratings) \
                    + "allfoldsaggregated_with_clusters_and_reviews_concatclusters.csv"




    if cfg.reviews:
        #aggregated per fold
        filepath_write_results_agg_perfold_withreviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                     + str(cfg.seed) + "_allfolds_ratingclass" + str(cfg.ratingclass) \
                                     + "_traningratings_" + str(cfg.whichratings) + "aggregatedperfold_withreviews.csv"
        #all folds agregated
        filepath_write_results_agg_allfolds_withreviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                     + str(cfg.seed) + "_allfold_ratingclass" + str(cfg.ratingclass) \
                                     + "_traningratings_" + str(cfg.whichratings) + "allfoldsaggregated_withreviews.csv"
    if cfg.clusters:
        # aggregated per fold
        filepath_write_results_agg_perfold_withreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                                                                    + "/evalaggregated_seed" \
                                             + str(cfg.seed) + "_allfolds_ratingclass" + str(cfg.ratingclass) \
                                             + "_traningratings_" + str(cfg.whichratings) \
                                            + "aggregatedperfold_withreviews_and_clusters.csv"
        # all folds agregated
        filepath_write_results_agg_allfolds_withreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed)\
                                                                     + "/evalaggregated_seed" \
                                              + str(cfg.seed) + "_allfold_ratingclass" + str(cfg.ratingclass) \
                                              + "_traningratings_" + str(cfg.whichratings) \
                                             + "allfoldsaggregated_withreviews_and_clusters.csv"
    if cfg.initialemb:
        # aggregated per fold
        filepath_write_results_agg_perfold_initial = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                        + str(cfg.seed) + "_allfolds_ratingclass" + str(cfg.ratingclass) \
                                        + "_traningratings_" + str(cfg.whichratings) + "aggregatedperfold.csv"
    # all folds agregated
    filepath_write_results_agg_allfolds_initial = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                        + str(cfg.seed) + "_aalfolds_ratingclass" + str(cfg.ratingclass) \
                                        + "_traningratings_" + str(cfg.whichratings) + "allfoldsaggregated.csv"




    for i in range(0, cfg.folds):


        if cfg.withreviewsandclusters:
            filepath_read_selected_with_reviews_and_clusters = []
            filepath_read_selected_euc_with_reviews_and_clusters = []
            filepath_read_selected_with_reviews_and_clusters_concatclusters = []
            filepath_read_selected_euc_with_reviews_and_clusters_concatclusters = []
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                f = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                    + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatreviewemb_withdotclusters.csv"
                filepath_read_selected_with_reviews_and_clusters.insert(c, f)
                feuc = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                       + "/evalset_euclidean_" + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                       str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                       + "clusters_withconcatreviewemb_withdotclusters.csv"
                filepath_read_selected_euc_with_reviews_and_clusters.insert(c, feuc)
                fconcat = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                    + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"
                filepath_read_selected_with_reviews_and_clusters_concatclusters.insert(c, fconcat)
                fconcat_euc = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                          + "euclidean_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                          + "traningratings_" + cfg.whichratings + str(c) \
                          + "clusters_withconcatreviewemb_withconcatclusters.csv"
                filepath_read_selected_euc_with_reviews_and_clusters_concatclusters.insert(c, fconcat_euc)



        if cfg.withclustersandreviews:
            filepath_read_selected_with_clusters_and_reviews = []
            filepath_read_selected_euc_with_clusters_and_reviews = []
            filepath_read_selected_with_clusters_and_reviews_concatclusters = []
            filepath_read_selected_euc_with_clusters_and_reviews_concatclusters = []
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                f = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                    + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_read_selected_with_clusters_and_reviews.insert(c, f)
                feuc = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                       + "/evalset_euclidean_" + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                       str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_read_selected_euc_with_clusters_and_reviews.insert(c, feuc)
                fconcat = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                    + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_read_selected_with_clusters_and_reviews_concatclusters.insert(c, fconcat)
                fconcat_euc = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                          + "euclidean_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                          + "traningratings_" + cfg.whichratings + str(c) \
                          + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_read_selected_euc_with_clusters_and_reviews_concatclusters.insert(c, fconcat_euc)

        if cfg.concatclusters:
            filepath_read_selected_withconcatclusters = []
            filepath_read_selected_euc_with_concatclusters = []
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                fcon = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                        + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                        + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"
                filepath_read_selected_withconcatclusters.insert(c, fcon)
                feuccon = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                        + "/evalset_euclidean_" + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                        str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"
                filepath_read_selected_euc_with_concatclusters.insert(c, feuccon)

        if cfg.clusters:
            filepath_read_selected_withclusters = []
            filepath_read_selected_euc_with_clusters = []
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                f = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" \
                        + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                        + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"
                filepath_read_selected_withclusters.insert(c, f)
                feuc = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                        + "/evalset_euclidean_" + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                        str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"
                filepath_read_selected_euc_with_clusters.insert(c, feuc)
        if cfg.reviews:
            filepath_read_selected_r = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" + "seed" \
                                     + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                     + "traningratings_" + cfg.whichratings + ".csv"

            filepath_read_selected_euc_r = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_euclidean_" \
                                         + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                                         str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + ".csv"
            filepath_read_selected_withreviews = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_" + "seed" \
                                     + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                     + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"

            filepath_read_selected_euc_with_reviews = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_euclidean_" \
                                         + "seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + \
                                         str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"
        if cfg.initialemb:
            filepath_read_selected_initial = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_"  + "seed" + str(cfg.seed) \
                             + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) + "traningratings_" \
                             + cfg.whichratings + ".csv"

            filepath_read_selected_euc_initial = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_euclidean_" + "seed" + str(cfg.seed) \
                                 + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) + "traningratings_" \
                                 + cfg.whichratings + ".csv"

        filepath_read_ImplMF = cfg.PredictionResultsDir + "baselines/implMF_fold" + str(i) + ".csv"
        filepath_read_UserKNN = cfg.PredictionResultsDir + "baselines/UserKNN_fold" + str(i) + ".csv"

        filepath_read_ground_truth = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/test-" + str(i) + "-one.csv"




        if cfg.withclustersandreviews:
            filepath_write_results_peruser_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_clusters_and_reviews.csv"
            filepath_write_results_agg_with_clusters_and_reviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_clusters_and_reviews.csv"
            filepath_write_results_peruser_with_clusters_and_reviews_concat = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_concatclusters_and_reviews.csv"
            filepath_write_results_agg_with_clusters_and_reviews_concat = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_concatclusters_and_reviews.csv"


        if cfg.withreviewsandclusters:
            filepath_write_results_peruser_with_reviews_and_clusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_reviews_and_clusters.csv"
            filepath_write_results_agg_with_reviews_and_clusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_reviews_and_clusters.csv"
            filepath_write_results_peruser_with_reviews_and_clusters_concat = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_reviews_and_concatclusters.csv"
            filepath_write_results_agg_with_reviews_and_clusters_concat = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_reviews_and_concatclusters.csv"



        if cfg.reviews:
            filepath_write_results_peruser_withclustersandreviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalperuser_seed" \
                                             + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                             + "_traningratings_" + str(cfg.whichratings) + "_with_reviews.csv"
            filepath_write_results_agg_withclustersandreviews = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                         + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                         + "_traningratings_" + str(cfg.whichratings) + "_with_reviews.csv"

        if cfg.clusters:
            filepath_write_results_peruser_witreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_clusters.csv"
            filepath_write_results_agg_withreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_clusters.csv"

        if cfg.concatclusters:
            filepath_write_results_peruser_witreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalperuser_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "with_concatclusters.csv"
            filepath_write_results_agg_withreviewsandclusters = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) \
                    + "/evalaggregated_seed" + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                    + "_traningratings_" + str(cfg.whichratings) + "_with_concatclusters.csv"

        if cfg.initialemb:
            filepath_write_results_peruser_initial = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalperuser_seed" \
                                     + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                     + "_traningratings_" + str(cfg.whichratings) + ".csv"
            filepath_write_results_agg_initial = cfg.EvaluationResultsDir + "seed" + str(cfg.seed) + "/evalaggregated_seed" \
                                     + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                     + "_traningratings_" + str(cfg.whichratings) + ".csv"

        #open ground eval files
        truth = pd.read_csv(filepath_read_ground_truth)
        #open predictions
        if cfg.initialemb:
            recommendationsl = pd.read_csv(filepath_read_selected_initial)
            recommendationsl['rank'] = recommendationsl['rank'].astype(int)

        if cfg.initialemb:
            recommendationsEUC = pd.read_csv(filepath_read_selected_euc_initial)
            recommendationsEUC['rank'] = recommendationsEUC['rank'].astype(int)


        recommendationsMF = pd.read_csv(filepath_read_ImplMF)
        recommendationsMF['rank'] = recommendationsMF['rank'].astype(int)

        recommendationsUserKNN = pd.read_csv(filepath_read_UserKNN)
        recommendationsUserKNN['rank'] = recommendationsUserKNN['rank'].astype(int)

        recommendations_withclusters_agg = pd.DataFrame()
        recommendations_withclusters_euc_agg = pd.DataFrame()
        recommendations_withconcatclusters_agg = pd.DataFrame()
        recommendations_withconcatclusters_euc_agg = pd.DataFrame()

        recommendations_with_clusters_and_reviews_agg = pd.DataFrame()
        recommendations_with_clusters_and_reviews_euc_agg = pd.DataFrame()
        recommendations_with_concatclusters_and_reviews_agg = pd.DataFrame()
        recommendations_with_concatclusters_and_reviews_euc_agg = pd.DataFrame()

        recommendations_with_reviews_and_clusters_agg = pd.DataFrame()
        recommendations_with_reviews_and_clusters_euc_agg = pd.DataFrame()
        recommendations_with_reviews_and_concatclusters_agg = pd.DataFrame()
        recommendations_with_reviews_and_concatclusters_euc_agg = pd.DataFrame()



        if cfg.withreviewsandclusters:
            # in range there is just one file array, may be a problem if those arrays are actually of different lengths
            for c in range(0, len(filepath_read_selected_with_reviews_and_clusters)):

                recommendations_with_reviews_and_clusters = pd.read_csv(filepath_read_selected_with_reviews_and_clusters[c])
                recommendations_with_reviews_and_clusters['rank'] = recommendations_with_reviews_and_clusters['rank'].astype(int)

                recommendations_with_reviews_and_clusters_agg = pd.concat([recommendations_with_reviews_and_clusters_agg, recommendations_with_reviews_and_clusters])
                recommendations_with_reviews_and_clusters_euc = pd.read_csv(filepath_read_selected_euc_with_reviews_and_clusters[c])
                recommendations_with_reviews_and_clusters_euc['rank'] = recommendations_with_reviews_and_clusters_euc['rank'].astype(int)

                recommendations_with_reviews_and_clusters_euc_agg = pd.concat([recommendations_with_reviews_and_clusters_euc_agg, recommendations_with_reviews_and_clusters_euc])


        if cfg.withclustersandreviews:
            for c in range(0, len(filepath_read_selected_with_clusters_and_reviews)):
                recommendations_with_concatclusters_and_reviews = \
                    pd.read_csv(filepath_read_selected_with_clusters_and_reviews_concatclusters[c])
                recommendations_with_concatclusters_and_reviews['rank'] = recommendations_with_concatclusters_and_reviews['rank'].astype(int)


                recommendations_with_concatclusters_and_reviews_agg = pd.concat([recommendations_with_concatclusters_and_reviews_agg, recommendations_with_concatclusters_and_reviews])

                recommendations_with_concatclusters_and_reviews_euc = \
                    pd.read_csv(filepath_read_selected_euc_with_clusters_and_reviews_concatclusters[c])
                recommendations_with_concatclusters_and_reviews_euc['rank'] = recommendations_with_concatclusters_and_reviews_euc['rank'].astype(int)
                recommendations_with_concatclusters_and_reviews_euc_agg = pd.concat([recommendations_with_concatclusters_and_reviews_euc_agg, recommendations_with_concatclusters_and_reviews_euc])
        if cfg.clusters:
            for c in range(0, len(filepath_read_selected_withclusters)):
                recommendations_withclusters = pd.read_csv(filepath_read_selected_withclusters[c])
                recommendations_withclusters['rank'] = recommendations_withclusters['rank'].astype(int)
                recommendations_withclusters_agg = pd.concat([recommendations_withclusters_agg, recommendations_withclusters])

                recommendations_withclusters_euc = pd.read_csv(filepath_read_selected_euc_with_clusters[c])
                recommendations_withclusters_euc['rank'] = recommendations_withclusters_euc['rank'].astype(int)

                recommendations_withclusters_euc_agg = pd.concat([recommendations_withclusters_euc_agg, recommendations_withclusters_euc])
        if cfg.concatclusters:
            for c in range(0, len(filepath_read_selected_withconcatclusters)):
                recommendations_withconcatclusters = pd.read_csv(filepath_read_selected_withconcatclusters[c])
                recommendations_withconcatclusters['rank'] = recommendations_withconcatclusters['rank'].astype(int)

                recommendations_withconcatclusters_agg = pd.concat([recommendations_withconcatclusters_agg, recommendations_withconcatclusters])

                recommendations_withconcatclusters_euc = pd.read_csv(filepath_read_selected_euc_with_concatclusters[c])
                recommendations_withconcatclusters_euc['rank'] = recommendations_withconcatclusters_euc['rank'].astype(int)


                recommendations_withconcatclusters_euc_agg = pd.concat([recommendations_withconcatclusters_euc_agg, recommendations_withconcatclusters_euc])

        if cfg.reviews:
            recommendationsl_withreviews = pd.read_csv(filepath_read_selected_withreviews)
            recommendationsl_withreviews['rank'] = recommendationsl_withreviews['rank'].astype(int)
            recommendationsEUC_withreviews = pd.read_csv(filepath_read_selected_euc_with_reviews)
            recommendationsEUC_withreviews['rank'] = recommendationsEUC_withreviews['rank'].astype(int)
        recs = []
        if cfg.initialemb:
            recs = [recommendationsl, recommendationsMF, recommendationsEUC, recommendationsUserKNN]


        if cfg.reviews:
            recs = recs + [recommendationsl_withreviews, recommendationsEUC_withreviews]
        if cfg.clusters:
            recs = recs + [recommendations_withclusters_agg, recommendations_withclusters_euc_agg]
        if cfg.concatclusters:
            recs = recs + [recommendations_withconcatclusters_agg, recommendations_withconcatclusters_euc_agg]
        if cfg.withclustersandreviews:
            recs = recs + \
                   [recommendations_with_clusters_and_reviews_agg, recommendations_with_clusters_and_reviews_euc_agg,
                    recommendations_with_concatclusters_and_reviews_agg, recommendations_with_concatclusters_and_reviews_euc_agg]
        if cfg.withreviewsandclusters:
            recs = recs + \
                   [recommendations_with_reviews_and_clusters_agg, recommendations_with_reviews_and_clusters_euc_agg,
                    recommendations_with_reviews_and_concatclusters_agg, recommendations_with_reviews_and_concatclusters_euc_agg]

        recommendations = pd.concat(recs, ignore_index=True, axis=0)


        #droping the rating column in truth file
        truth.drop("rating", axis=1, inplace=True)



        #evaluations analysis

        rla = topn.RecListAnalysis()
        rla.add_metric(topn.ndcg, name="ndcg_5", k=5)
        rla.add_metric(topn.hit, name="hit_5", k=5)
        rla.add_metric(topn.precision, name="P_5", k=5)
        rla.add_metric(topn.recall, name="R_5", k=5)
        rla.add_metric(topn.recip_rank, name="RR_5", k=5)
        rla.add_metric(topn.ndcg, name="ndcg_10", k=10)
        rla.add_metric(topn.hit, name="hit_10", k=10)
        rla.add_metric(topn.precision, name="P_10", k=10)
        rla.add_metric(topn.recall, name="R_10", k=10)
        rla.add_metric(topn.recip_rank, name="RR_10", k=10)
        rla.add_metric(topn.ndcg, name="ndcg_20", k=20)
        rla.add_metric(topn.hit, name="hit_20", k=20)
        rla.add_metric(topn.precision, name="P_20", k=20)
        rla.add_metric(topn.recall, name="R_20", k=20)
        rla.add_metric(topn.recip_rank, name="RR_20", k=20)
        rla.add_metric(topn.ndcg, name="ndcg_50", k=50)
        rla.add_metric(topn.hit, name="hit_50", k=50)
        rla.add_metric(topn.precision, name="P_50", k=50)
        rla.add_metric(topn.recall, name="R_50", k=50)
        rla.add_metric(topn.recip_rank, name="RR_50", k=50)
        rla.add_metric(topn.ndcg, name="ndcg_100", k=100)
        rla.add_metric(topn.hit, name="hit_100", k=100)
        rla.add_metric(topn.precision, name="P_100", k=100)
        rla.add_metric(topn.recall, name="R_100", k=100)
        rla.add_metric(topn.recip_rank, name="RR_100", k=100)
        rla.add_metric(topn.ndcg, name="ndcg_un")
        rla.add_metric(topn.hit, name="hit_un")
        rla.add_metric(topn.precision, name="P_un")
        rla.add_metric(topn.recall, name="R_un")
        rla.add_metric(topn.recip_rank, name="RR_un")



        results = rla.compute(recommendations, truth)


        results_agg = results.groupby("Algorithm").mean()
        #this should be here for the evaluation of the whole method
        #results.to_csv(filepath_write_results_peruser_with_reviews_and_clusters_concat)

        # this should be here for evaluation for impact of reviews only
        #results.to_csv(filepath_write_results_peruser_initial)

        #This should be here for the evaluation with the whole method
        #results_agg.to_csv(filepath_write_results_agg_with_reviews_and_clusters_concat)

        # this should be here for evaluation for impact of reviews only
        results_agg.to_csv(filepath_write_results_agg_initial)
        results_mean = pd.concat([results_mean, results])
        results_mean_agg = pd.concat([results_mean_agg, results_agg])
        #print("Time of finish: " + str(dt.datetime.now()))

    print(results_mean_agg)
    #this should be here for evaluation of impact of reviews only
    #results_mean_agg.to_csv(filepath_write_results_agg_perfold_initial)

    #this should be here for the whole method
    #results_mean_agg.to_csv(filepath_write_results_agg_perfold_with_reviews_and_clusters_concat)

    results_all_agg_mean = results_mean_agg.groupby("Algorithm").mean()
    #this should be here for the evaluation of the whole method
    #results_all_agg_mean.to_csv(filepath_write_results_agg_allfolds_with_reviews_and_clusters_concat)

    #this should be here for evaluation of the impact of the reviews only
    results_all_agg_mean.to_csv(filepath_write_results_agg_allfolds_initial)
    end = dt.datetime.now()

    print("Time of end: " + str(end))
    print("Duration: " + str(end - start))


def main():
    run()

if __name__ == "__main__":
    main()