
# Peter Dolog, Aalborg University, dolog@cs.aau.dk
# it requires item and user embeddings
# it assumes user and item id in the first column
# each user has exactly as many rows as classes of ratings
# example: if we have rating 1 for like, 0 for do not know, and -1 for dislike, they are expected in this order in the file
# the script produces recommendations with normalized dot products
# then it formats it into the format required for evaluation: item, user, score, rank, algorithm
# it is required to run it per class - i.e. each class has separate embeddings for items and users
# files can be created from a single embedding file by running joining.py utility
# for k-fold it is required to run 5 times for different folds separately

import pandas as pd
import numpy as np
import datetime as dt
import config as cfg




def run():
    #filepath_read = "Data/movies_users232.csv"
    #filepath_write_dot = "Data/dotproducts232.csv"


    #clusters = cfg.clusters
    #dimmensionality = cfg.dimmensionality
    #operation = "dot"
    start = dt.datetime.now()
    print("Time of start: " + str(start))

    print(cfg.whichratings)

    if cfg.withreviewsandclusters:
        for f in range(0, cfg.folds):
            print("fold nr.: " + str(f))
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                print("clusters: " + str(c))

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


                # write dot clusters predictions
                filepath_write_dot_filtered_reviews_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) \
                                          + "clusters_withconcatreviewemb_withdotclusters.csv"

                filepath_write_dot_eval_filtered_reviews_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                                               + "/evalset_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) \
                                               + "clusters_withconcatreviewemb_withdotclusters.csv"

                # write concat clusters predictions
                filepath_write_dot_filtered_reviews_withdotclusters_concat = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                        + "/dotproduct_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"

                filepath_write_dot_eval_filtered_reviews_withdotclusters_concat = cfg.PredictionResultsDir + "seed" \
                    + str(cfg.seed) + "/evalset_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                                        + "clusters_withconcatreviewemb_withconcatclusters.csv"




                # dot clusters
                #-------------
                users = pd.read_csv(filepath_read_users_reviewsfirst_with_clusters_dot, header=None)
                items = pd.read_csv(filepath_read_filtermovies_reviewsfirst_withclusters_dot,  index_col=False, header=None)
                #extracting user id and making a dictionary for later use
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                #extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

                #computing normalized dot product

                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy/unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy/inorm

                items_normalizedT = items_normalized.transpose()

                dp = users_normalized.dot(items_normalizedT)

                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                #formating to evaluation format required
                df = pd.DataFrame()
                preference = 0

                lastcolumn = len(dpt.columns) - 1

                create_recommendation_list_for_eval(df, dpt,
                                                    filepath_write_dot_eval_filtered_reviews_withdotclusters,
                                                    "dotproduct_reviews_and_dotclusters" + str(c) + "clusters")

                # concat clusters
                # -------------
                users = pd.read_csv(filepath_read_users_reviewsfirst_with_clusters_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_reviewsfirst_withclusters_concat, index_col=False, header=None)
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                # extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy / unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)


                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy / inorm

                items_normalizedT = items_normalized.transpose()

                dp = users_normalized.dot(items_normalizedT)

                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                # formating to evaluation format required
                df = pd.DataFrame()
                preference = 0

                lastcolumn = len(dpt.columns) - 1
                create_recommendation_list_for_eval(df, dpt,
                                                    filepath_write_dot_eval_filtered_reviews_withdotclusters_concat,
                                                    "dotproduct_reviews_and_concatclusters_" + str(c) + "clusters")


    if cfg.withclustersandreviews:
        for f in range(0, cfg.folds):
            print("fold nr.: " + str(f))
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                print("clusters: " + str(c))

                #read dot clusters
                filepath_read_users_with_clusters_dot_reviews = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                         "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                         str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                                                         + "row_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_read_filtermovies_withclusters_dot_reviews = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                               "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                               str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                               "_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                # read concat clusters
                filepath_read_users_with_clusters_dot_reviews_concat = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                            "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                            str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                            + "row_" + str(c) + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_read_filtermovies_withclusters_dot_reviews_concat = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) \
                                            + "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" \
                                            + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                            "_" + str(c) + "clusters_withconcatclusters_withconcatreviewemb.csv"

                # write dot clusters predictions
                filepath_write_dot_filtered_withdotclusters_reviews = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"

                filepath_write_dot_eval_filtered_withdotclusters_reviews = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"

                # write concat clusters predictions
                filepath_write_dot_filtered_withdotclusters_reviews_concat = cfg.PredictionResultsDir + "seed" + str(cfg.seed) \
                        + "/dotproduct_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                    + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters_withconcatreviewemb.csv"

                filepath_write_dot_eval_filtered_withdotclusters_reviews_concat = cfg.PredictionResultsDir + "seed" \
                    + str(cfg.seed) + "/evalset_seed" + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" \
                    + str(cfg.ratingclass) + "traningratings_" + cfg.whichratings + str(c) \
                                        + "clusters_withconcatclusters_withconcatreviewemb.csv"


                # dot clusters
                #-------------
                users = pd.read_csv(filepath_read_users_with_clusters_dot_reviews, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot_reviews,  index_col=False, header=None)
                print("Items:")
                print(items)
                print("Users:")
                print(users)
                #extracting user id and making a dictionary for later use
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                #extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

                #computing normalized dot product

                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy/unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy/inorm

                items_normalizedT = items_normalized.transpose()

                print("users:")
                print(users_normalized)
                print("items:")
                print(items_normalizedT)

                dp = users_normalized.dot(items_normalizedT)

                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                #formating to evaluation format required
                df = pd.DataFrame()
                preference = 0

                lastcolumn = len(dpt.columns) - 1
                create_recommendation_list_for_eval(df, dpt,
                                                    filepath_write_dot_eval_filtered_withdotclusters_reviews,
                                                    "dotproduct_dotclusters_and_reviews_" + str(c) + "clusters")

                # concat clusters
                # -------------
                users = pd.read_csv(filepath_read_users_with_clusters_dot_reviews_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot_reviews_concat, index_col=False, header=None)
                # extracting user id and making a dictionary for later use
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                # extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

                # computing normalized dot product

                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy / unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)


                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy / inorm

                items_normalizedT = items_normalized.transpose()

                dp = users_normalized.dot(items_normalizedT)

                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                # formating to evaluation format required
                df = pd.DataFrame()
                preference = 0

                lastcolumn = len(dpt.columns) - 1
                create_recommendation_list_for_eval(df, dpt, filepath_write_dot_eval_filtered_withdotclusters_reviews_concat,
                                                    "dotproduct_concatclusters_and_reviews_" + str(c) + "clusters")

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

                filepath_write_dot_filtered_withconcatclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"
                filepath_write_dot_eval_filtered_withconcatclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) + "clusters_withconcatclusters.csv"






                users = pd.read_csv(filepath_read_users_with_clusters_concat, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_concat,  index_col=False, header=None)
                userid = users.iloc[:, 0]


                #extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)


                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy/unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)


                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy/inorm

                items_normalizedT = items_normalized.transpose()

                dp = users_normalized.dot(items_normalizedT)


                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                df = pd.DataFrame()
                preference = 0
                lastcolumn = len(dpt.columns)-1

                create_recommendation_list_for_eval(df, dpt, filepath_write_dot_eval_filtered_withconcatclusters,
                                                    "dotproduct_concatclusters_" + str(c) + "clusters")

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

                filepath_write_dot_filtered_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                          + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"
                filepath_write_dot_eval_filtered_withdotclusters = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_seed" \
                                               + str(cfg.seed) + "_fold" + str(f) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + str(c) + "clusters_withdotclusters.csv"






                users = pd.read_csv(filepath_read_users_with_clusters_dot, header=None)
                items = pd.read_csv(filepath_read_filtermovies_withclusters_dot,  index_col=False, header=None)
                print("Items:")
                print(items)
                print("Users:")
                print(users)
                #extracting user id and making a dictionary for later use
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                #extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()

                users_copy = users.copy(deep=True)
                users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

                print(users_copy)
                #computing normalized dot product

                unorm = np.linalg.norm(users_copy)
                users_normalized = users_copy/unorm

                items_copy = items.copy(deep=True)
                items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

                print(items_copy)

                inorm = np.linalg.norm(items_copy)
                items_normalized = items_copy/inorm

                items_normalizedT = items_normalized.transpose()

                dp = users_normalized.dot(items_normalizedT)


                #formating to evaluation format required

                dpt = dp.T
                dpt[len(dpt.columns)] = itemid
                # formating to evaluation format required
                df = pd.DataFrame()
                lastcolumn = len(dpt.columns) - 1
                create_recommendation_list_for_eval(df, dpt, filepath_write_dot_eval_filtered_withdotclusters, "dotproduct_dotclusters_" + str(c) + "clusters")

    elif cfg.initialemb:
        for i in range(0, cfg.folds):

            # directory where original embeddings is
            filepath_read_users = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                                  + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row.csv"
            filepath_read_items = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                                  + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + ".csv"


            filepath_write_dot_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                          + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                          + "traningratings_" + cfg.whichratings + ".csv"
            filepath_write_dot_eval_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_seed" \
                                               + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                               + "traningratings_" + cfg.whichratings + ".csv"


            users = pd.read_csv(filepath_read_users, header=None)
            items = pd.read_csv(filepath_read_items,  index_col=False, header=None)
            print("Items:")
            print(items)
            print("Users:")
            print(users)
            #extracting user id and making a dictionary for later use
            userid = users.iloc[:, 0]


            #extracting itemid and making a dictionary for later use
            itemid = items.iloc[:, 0]


            users_copy = users.copy(deep=True)
            users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

            #computing normalized dot product

            unorm = np.linalg.norm(users_copy)
            users_normalized = users_copy/unorm
            print("USers Normalized:")
            print(users_normalized)

            items_copy = items.copy(deep=True)
            items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

            print(items_copy)
            #exit()
            print("Creating dot product!")
            inorm = np.linalg.norm(items_copy)
            items_normalized = items_copy/inorm

            items_normalizedT = items_normalized.transpose()

            dp = users_normalized.dot(items_normalizedT)
            print(dp)
            dpt = dp.T
            dpt[len(dpt.columns)] = itemid
            #formating to evaluation format required
            df = pd.DataFrame()
            preference = 0
            print("Creating list for recommendation evaluation!")
            create_recommendation_list_for_eval(df, dpt, filepath_write_dot_eval_filtered, "dotproduct")
            lastcolumn = len(dpt.columns) - 1

    elif cfg.reviews:
        for i in range(0, cfg.folds):

            filepath_read_users = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                              + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row_withconcatreviewemb.csv"
            filepath_read_items = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                              + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + "_withconcatreviewemb.csv"




            filepath_write_dot_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/dotproduct_seed" \
                                       + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                       + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"
            filepath_write_dot_eval_filtered = cfg.PredictionResultsDir + "seed" + str(cfg.seed) + "/evalset_seed" \
                                      + str(cfg.seed) + "_fold" + str(i) + "_ratingclass" + str(cfg.ratingclass) \
                                      + "traningratings_" + cfg.whichratings + "_withconcatreviewemb.csv"



            users = pd.read_csv(filepath_read_users, header=None)
            items = pd.read_csv(filepath_read_items,  index_col=False, header=None)
            print("Items:")
            print(items)
            print("Users:")
            print(users)
            #extracting user id and making a dictionary for later use
            userid = users.iloc[:, 0]


            #extracting itemid and making a dictionary for later use
            itemid = items.iloc[:, 0]

            users_copy = users.copy(deep=True)
            users_copy.drop(users_copy.columns[[0]], axis=1, inplace=True)

            #computing normalized dot product

            unorm = np.linalg.norm(users_copy)
            users_normalized = users_copy/unorm

            items_copy = items.copy(deep=True)
            items_copy = items_copy.drop(items_copy.columns[[0]], axis=1)

            print(items_copy)

            inorm = np.linalg.norm(items_copy)
            items_normalized = items_copy/inorm

            items_normalizedT = items_normalized.transpose()

            dp = users_normalized.dot(items_normalizedT)

            dpt = dp.T
            dpt[len(dpt.columns)] = itemid
            #formating to evaluation format required
            df = pd.DataFrame()
            preference = 0

            lastcolumn = len(dpt.columns) - 1

            create_recommendation_list_for_eval(df, dpt, filepath_write_dot_eval_filtered, "dotproduct_reviews")


    end = dt.datetime.now()
    print("Time of end: " + str(end))
    print("Elapsed time: " + str(end - start))


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
        candidatelist.to_csv(path, index=False, mode='a', header=False)
    print("Finished Creating list!")
    #return df


def main():
    run()

if __name__ == "__main__":
    main()