# Peter Dolog, Aalborg University, dolog@cs.aau.dk
# it requires item and user embeddings
# it assumes user and item id in the first column
# each user has exactly as many rows as classes of ratings
# example: if we have rating 1 for like, 0 for do not know, and -1 for dislike, they are expected in this order in the file
# the script produces recommendations with normalized euclidean distance
# then it formats it into the format required for evaluation: item, user, score, rank, algorithm
# it is required to run it per class - i.e. each class has separate embeddings for items and users
# files can be created from a single embedding file by running joining.py utility
# for k-fold it is required to run 5 times for different folds separately

import pandas as pd
import numpy as np
import datetime as dt
from scipy.spatial import distance
import config as cfg



def run():

    start = dt.datetime.now()

    print("Time of start: " + str(start))


    if cfg.withreviewsandclusters:
        for f in range(0, cfg.folds):
            print("fold nr.: " + str(f))
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                print("clusters: " + str(c))

                # read dot clusters for reviews first concatenated

                filepath_read_filtermovies_reviewsfirst_withclusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" \
                            + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                            str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                            "_" + str(c) + "clusters_withconcatreviewemb_withdotclusters.csv"

                filepath_read_users_reviewsfirst_with_clusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" \
                            + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                            str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                            + "row_" + str(c) + "clusters_withconcatreviewemb_withdotclusters.csv"
                # read concat clusters for reviews first concatenated
                filepath_read_filtermovies_reviewsfirst_withclusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" \
                        + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                        str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                        "_" + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"

                filepath_read_users_reviewsfirst_with_clusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" \
                        + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                        str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                        "row_" + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"


                # dot cluster write files (prediction and transformation for evaluation)
                filepath_write_euc_filtered_withdotclusters_with_reviews_and_clusters = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatreviewemb_withdotclusters.csv"
                filepath_write_euc_eval_filtered_withdotclusters_reviews_and_clusters = cfg.PredictionResultsDir + "seed" \
                    + str(cfg.seed) + "/evalset_euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatreviewemb_withdotclusters.csv"

                # concat cluster write files (prediction and transformation for evaluation)
                filepath_write_euc_filtered_withdotclusters_with_reviews_and_clusters_concat = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatreviewemb_withconcatclusters.csv"
                filepath_write_euc_eval_filtered_withdotclusters_reviews_and_clusters_concat = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/evalset_euclidean_seed" + str(cfg.seed) + "_fold" + str(f) \
                    + "_ratingclass" + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatreviewemb_withconcatclusters.csv"

                #prediction for dot clusters
                #---------------------------
                users = pd.read_csv(filepath_read_users_reviewsfirst_with_clusters_dot, header=None)
                items = pd.read_csv(filepath_read_filtermovies_reviewsfirst_withclusters_dot, index_col=False, header=None)



                # extracting user id
                userid = users.iloc[:, 0]


                # extracting item id

                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()

                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()
                #preference = 0

                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt,
                                                    filepath_write_euc_eval_filtered_withdotclusters_reviews_and_clusters,
                                                    "euclidean_reviews_and_dotclusters_" + str(c) + "clusters")

                # prediction for concat clusters
                # ---------------------------
                users = pd.read_csv(filepath_read_users_reviewsfirst_with_clusters_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_reviewsfirst_withclusters_concat, index_col=False, header=None)

                # extracting user id
                userid = users.iloc[:, 0]


                # extrcting item id

                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()

                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()

                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt,
                                                    filepath_write_euc_eval_filtered_withdotclusters_reviews_and_clusters_concat,
                                                    "euclidean_reviews_and_concatclusters_" + str(c) + "clusters")



    if cfg.withclustersandreviews:
        for f in range(0, cfg.folds):
            print("fold nr.: " + str(f))
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                print("clusters: " + str(c))
                # dot clusters read file (users and items embeddings)
                filepath_read_users_with_clusters_dot_reviews = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                    "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + str(cfg.seed) + "filtered" \
                    + str(cfg.ratingclass) + "row_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_read_filtermovies_withclusters_dot_reviews = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                    "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + str(cfg.seed) + "filteredclass" \
                    + str(cfg.ratingclass) + "_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"

                # concat clusters read files (users and items)
                filepath_read_users_with_clusters_dot_reviews_concat = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                        "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + str(cfg.seed) + "filtered" \
                        + str(cfg.ratingclass) + "row_" + str(c) + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_read_filtermovies_withclusters_dot_reviews_concat = cfg.WithReviwesEmbeddingsDir + "seed" \
                    + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + str(cfg.seed) \
                    + "filteredclass" + str(cfg.ratingclass) + "_" + str(c) \
                    + "clusters_withconcatclusters_withconcatreviewemb.csv"

                # dot cluster write files (prediction and transformation for evaluation)
                filepath_write_euc_filtered_withdotclusters_with_culsters_and_reviews = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_write_euc_eval_filtered_withdotclusters_clusters_and_reviews = cfg.PredictionResultsDir + "seed" \
                    + str(cfg.seed) + "/evalset_euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withdotclusters_withconcatreviewemb.csv"

                # concat cluster write files (prediction and transformation for evaluation)
                filepath_write_euc_filtered_withdotclusters_with_culsters_and_reviews_concat = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/euclidean_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_write_euc_eval_filtered_withdotclusters_clusters_and_reviews_concat = cfg.PredictionResultsDir \
                    + "seed" + str(cfg.seed) + "/evalset_euclidean_seed" + str(cfg.seed) + "_fold" + str(f) \
                    + "_ratingclass" + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                    + "clusters_withconcatclusters_withconcatreviewemb.csv"

                #prediction for dot clusters
                #---------------------------
                users = pd.read_csv(filepath_read_users_with_clusters_dot_reviews, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot_reviews, index_col=False, header=None)


                # extracting user id
                userid = users.iloc[:, 0]

                # extracting item id

                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()

                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()

                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt,
                                                    filepath_write_euc_eval_filtered_withdotclusters_clusters_and_reviews,
                                                    "euclidean_dotclusters_and_reviews_" + str(c) + "clusters")

                # prediction for concat clusters
                # ---------------------------
                users = pd.read_csv(filepath_read_users_with_clusters_dot_reviews_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot_reviews_concat, index_col=False, header=None)


                # extracting user id
                userid = users.iloc[:, 0]

                # extracting item id

                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)



                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()

                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()

                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt, filepath_write_euc_eval_filtered_withdotclusters_clusters_and_reviews_concat,
                                                    "euclidean_concatclusters_and_reviews_" + str(c) + "clusters")

    elif cfg.clusters:
        for f in range(0, cfg.folds):
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                filepath_read_users_with_clusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                         "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                         str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                                                         + "row_" + str(c) + "clusters_withdotclusters.csv"
                filepath_read_filtermovies_withclusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                               "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                               str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                               "_" + str(c) + "clusters_withdotclusters.csv"

                filepath_write_euc_filtered_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/euclidean_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"
                filepath_write_euc_eval_filtered_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                                                                             + "/evalset_euclidean_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"


                users = pd.read_csv(filepath_read_users_with_clusters_dot, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot , index_col=False, header=None)


                # extracting user id
                userid = users.iloc[:, 0]


                # extracting item id

                itemid = items.iloc[:, 0]


                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)



                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()


                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()
                preference = 0

                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt, filepath_write_euc_eval_filtered_withdotclusters,
                                                    "euclidean_dotclusters_" + str(c) + "clusters")

    elif cfg.concatclusters:
        for f in range(0, cfg.folds):
            print("Fold nr.: " + str(f))
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                print("Nr. of clusters: " + str(c))
                filepath_read_users_with_clusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                    "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                    str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                    + "row_" + str(c) + "clusters_withconcatclusters.csv"
                filepath_read_filtermovies_withclusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                            "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                            str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                            "_" + str(c) + "clusters_withconcatclusters.csv"

                filepath_write_euc_filtered_withconcatclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/euclidean_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"
                filepath_write_euc_eval_filtered_withconcatclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                                                + "/evalset_euclidean_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"


                users = pd.read_csv(filepath_read_users_with_clusters_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_concat, index_col=False, header=None)


                # extracting user id
                userid = users.iloc[:, 0]


                # extracting item id

                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                # computing euclidean
                eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

                eud = eud.transpose()

                eudt = eud.T
                eudt[len(eudt.columns)] = itemid
                # formating to the format required for evaluation
                df = pd.DataFrame()
                preference = 0
                lastcolumn = len(eudt.columns) - 1
                create_recommendation_list_for_eval(df, eudt, filepath_write_euc_eval_filtered_withconcatclusters, "euclidean_concatclusters_" + str(c) + "clusters")

    elif cfg.initialemb:
        for i in range(0, cfg.folds):


            filepath_read_users = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                              + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row.csv"
            filepath_read_items = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                              + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + ".csv"


            filepath_write_euc_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/euclidean_seed" \
                                      + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                      + "traningratings_" + cfg.whichratings + ".csv"
            filepath_write_euc_eval_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_euclidean_seed" \
                                           + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                           + "traningratings_" + cfg.whichratings + ".csv"

            users = pd.read_csv(filepath_read_users, header=None)
            items = pd.read_csv(filepath_read_items, index_col=False, header=None)


            #extracting user id
            userid = users.iloc[:, 0]



            #extracting item id

            itemid = items.iloc[:, 0]

            users_copy = users.copy(deep=True)
            users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)



            items_copy = items.copy(deep=True)
            items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)


            #computing euclidean
            eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

            eud = eud.transpose()


            eudt = eud.T
            eudt[len(eudt.columns)] = itemid
            #formating to the format required for evaluation
            df = pd.DataFrame()
            preference = 0
            lastcolumn = len(eudt.columns) - 1
            create_recommendation_list_for_eval(df, eudt, filepath_write_euc_eval_filtered, "euclidean")


    elif cfg.reviews:
        for i in range(0, cfg.folds):

            filepath_read_users = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/userEmb_" \
                                 + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) \
                                  + "filtered" + str(cfg.ratingclass) + "row_withconcatreviewemb.csv"
            filepath_read_items = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/movieEmb_" \
                                + cfg.whichratings + "_train" \
                                  + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) \
                                  + "_withconcatreviewemb.csv"
            filepath_write_euc_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/euclidean_seed" \
                                          + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"
            filepath_write_euc_eval_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_euclidean_seed" \
                                               + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"

            users = pd.read_csv(filepath_read_users, header=None)
            items = pd.read_csv(filepath_read_items, index_col=False, header=None)


            #extracting user id
            userid = users.iloc[:, 0]



            #extracting item id

            itemid = items.iloc[:, 0]

            users_copy = users.copy(deep=True)
            users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


            items_copy = items.copy(deep=True)
            items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)


            #computing euclidean
            eud = pd.DataFrame(distance.cdist(items_copy, users_copy, metric='euclidean'))

            eud = eud.transpose()

            eudt = eud.T
            eudt[len(eudt.columns)] = itemid
            #formating to the format required for evaluation
            df = pd.DataFrame()
            preference = 0
            lastcolumn = len(eudt.columns) - 1
            create_recommendation_list_for_eval(df, eudt, filepath_write_euc_eval_filtered, "euclidean_reviews")


    end = dt.datetime.now()


    print("Time of end: " + str(end))
    print("Duration: " + str(end - start))

def create_recommendation_list_for_eval(df, dpt, path, algorithm):
    lastcolumn = len(dpt.columns) - 1
    candidatelist = pd.DataFrame(columns = ['item', 'score', 'user', 'rank', 'Algorithm'])
    candidatelist.to_csv(path, index=False)
    for index in range(dpt.shape[1] - 1):
        rowdf = dpt.iloc[:, [index, lastcolumn]]
        rowdf.columns = ['score', 'item']
        columns_titles = ["item", "score"]
        rowdf = rowdf.reindex(columns=columns_titles)
        rowdf = rowdf.sort_values(by=['score'], ascending=False)
        rowdf = rowdf.reset_index(drop=True)
        rowdf["user"] = index
        rowdf.insert(3, "rank", np.arange(1, rowdf.shape[0] + 1))
        rowdf["Algorithm"] = algorithm
        candidatelist = rowdf.iloc[:cfg.recommendationlistsize]
        # rowdf.to_csv(path, index=False, mode='a')
        candidatelist.to_csv(path, index=False, mode='a', header=False)
    print("Finished Creating list!")
    #return df


def main():
    run()

if __name__ == "__main__":
    main()