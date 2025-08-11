#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#it requires itemembeddings, json mappings of movie ids and URLs and sentence embeddings
#it requires user embeddings and user-item ratings
#it produces a csv file with id and url mappings transformed from json
#it produces a csv file with items embeddings and movie URI
#it traverses the items embeddings and for each movie queries sentence of reviews
#embeddings. If exists, it aggregates it to mean.
#it adds the concatenates the aggregated review embeddings to corresponding item embeddings
#it saves such concatenated embbeding to a csv file.
#it looks for items rated by users and aggregates their review embeddings (in train folds)
#it concatenates such an aggregated embeddings to user embeddings

# The review embeddings for items and users are aggregated beforehand form amazon and yelp datasets. On mindreader, remapping with url is needed before, and therefore we have 2 different functions
# Function for amazon is also used for yelp dataset



# improvement on mind reader: to improve speed and scalability, refactor the integration with finding and aggregating
# sentence embeddings in the same way as for the amazon dataset. See amazonsentenceemb.py

import pandas as pd
import joblib as jb
import numpy as np
import json as js
import datetime as dt



from typing import List, Union
import os
import config as cfg

#load embeddings of reviews for URLs - relevant for MindReader dataset and how embedding was made
def load_embeddings(sentence_emb_dir: str, movie_uris: Union[str, List[str]], split="train"):
    if isinstance(movie_uris, str):
        movie_uris = [movie_uris]

    df = pd.read_csv(os.path.join(sentence_emb_dir, f"{split}_df_with_uris.csv"), sep=";")
    sentence_embs = jb.load(os.path.join(sentence_emb_dir, f"{split}_embeddings.pbz2"))

    id_df = df[df["ml_uri"].isin(movie_uris)]

    embs = sentence_embs[id_df["id"].values, :]
    return embs







def run_on_mindreader():
    sentenceembeddings = cfg.SentenceEmbDir + cfg.SentenceEmbFile
    jsonurisfile = cfg.SentenceEmbDir + cfg.jsonURIfile



    sentencedata = jb.load(sentenceembeddings)





    sent_dir = cfg.SentenceEmbDir


    # load data using Python JSON module
    with open(jsonurisfile,'r') as f:
        data = js.loads(f.read())
    # Flatten data
    df_nested_list = pd.json_normalize(data)




    jsonTrans = df_nested_list.transpose()
    jsonTrans.reset_index(inplace=True)
    jsonTrans = jsonTrans.rename(columns = {'index':'id'})
    jsonTrans = jsonTrans.rename(columns = {0:'URL'})

    start = dt.datetime.now()
    print("Time of start: " + str(start))


    if cfg.withclustersandreviews:
        for i in range(0, cfg.folds):
            print("Fold: " + str(i))
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                print("Nr. of clusters: " + str(c))
                meantime = dt.datetime.now()
                print("Current time: " + str(meantime))
                print("Elapsed time so far: " + str(meantime - start))


                filepath_read_users = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + "/userEmb_" \
                        + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filtered" \
                        + str(cfg.ratingclass) + "row_" + str(c) + "clusters_withdotclusters.csv"
                filepath_read_items = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + "/movieEmb_" \
                        + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filteredclass" \
                        + str(cfg.ratingclass) + "_" + str(c) + "clusters_withdotclusters.csv"

                filepath_read_users_concatclusters = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) \
                                                     + "/userEmb_" + cfg.whichratings + "_train" \
                                      + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row_" \
                                                     + str(c) + "clusters_withconcatclusters.csv"
                filepath_read_items_concatclusters = cfg.EmbeddingsFolderWithClusters + "seed" + str(
                    cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                                      + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) \
                                                     + "_" + str(c) + "clusters_withconcatclusters.csv"

                filepath_write_filtermovies_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                    "_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_write_filterusers_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                    "row_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"

                filepath_write_filtermovies_withreviewemb_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                                                          + str(cfg.seed) + \
                                                            "/movieEmb_" + cfg.whichratings + "_train" + str(i) \
                                                            + "_seed" + str(cfg.seed) + "filteredclass" \
                                                                          + str(cfg.ratingclass) + "_" + str(c) \
                                                            + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_write_filterusers_withreviewemb_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                                        + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                                                        + str(i) + "_seed" + str(cfg.seed) + "filtered" \
                                                        + str(cfg.ratingclass) + "row_" + str(c) \
                                                            + "clusters_withconcatclusters_withconcatreviewemb.csv"

                filepath_write_filtermovies_withreviewurls = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                   "_" + str(c) + "clusters_withdotclusters_withconcatreviewurl.csv"

                filepath_write_filterusers_withreviewurls = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                            "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                            str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                            "row_withcontactreviewurl.csv""row_" + "_" + str(c) \
                                                            + "clusters_withdotclusters_withconcatreviewurl.csv"

                filepath_write_filtermovies_withreviewurls_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                                + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" + str(i) \
                                                + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) \
                                                + "_" + str(c) + "clusters_withconcatclusters_withconcatreviewurl.csv"

                filepath_write_filterusers_withreviewurls_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                        + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" + str(i) \
                                        + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                        "row_withcontactreviewurl.csv""row_" + "_" + str(c) \
                                        + "clusters_withconcatclusters_withconcatreviewurl.csv"

                filepath_read_user_train_ratings = cfg.FoldsDir + "test-" + str(i) + ".csv"

                #training set
                user_train_ratings = pd.read_csv(filepath_read_user_train_ratings)

                # dot clusters
                # -----------------
                # embeddings of users, items and train file
                users = pd.read_csv(filepath_read_users, header=None)
                items = pd.read_csv(filepath_read_items, index_col=False, header=None)

                # extracting user id and making a dictionary for later use
                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()

                # extracting itemid and making a dictionary for later use
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()


                items = items.rename(columns={0: 'id'})


                items['id'] = items['id'].astype(int)
                jsonTrans['id'] = jsonTrans['id'].astype(int)


                itemswithurl = pd.merge(items, jsonTrans, how="left", on="id").fillna(0)




                itemswithurl.to_csv(filepath_write_filtermovies_withreviewurls)


                meanreviewsemb = pd.DataFrame()
                for index, row in itemswithurl.iterrows():
                    # download sentence embeddings folder and run this with e.g.
                    movieuri = row['URL']
                    if movieuri != 0:
                        embs = load_embeddings(sent_dir, movieuri, split="train")

                    if embs.size > 0:
                        reviews = pd.DataFrame(embs)
                        meanreview = pd.DataFrame(reviews.aggregate('mean')).T
                    else:
                        reviews = pd.DataFrame(np.zeros((1, 768)))
                        meanreview = reviews
                        # reviews = pd.DataFrame()
                    meanreview['URL'] = movieuri
                    meanreview['id'] = row['id']
                    meanreviewsemb = pd.concat([meanreviewsemb, meanreview], ignore_index=True)


                itemswithreviewembs = pd.merge(itemswithurl, meanreviewsemb, how="left", on="id").fillna(0)

                itemswithreviewembs = itemswithreviewembs.drop(['URL_x', 'URL_y'], axis=1)
                itemswithreviewembs.to_csv(filepath_write_filtermovies_withreviewemb, index=False, header=False)
                meanuserreviewemb = pd.DataFrame()
                for index, row in users.iterrows():
                    userid = row[0]
                    ratedmovies = user_train_ratings.loc[user_train_ratings['user'] == userid]
                    userreviewembeddings = pd.merge(ratedmovies, meanreviewsemb, how="inner", left_on="item", right_on="id")
                    userreviewembeddings = userreviewembeddings.drop(['item', 'URL', 'id', 'rating'], axis=1)
                    meanuserreview = pd.DataFrame(userreviewembeddings.aggregate('mean')).T
                    meanuserreviewemb = pd.concat([meanuserreviewemb, meanuserreview], ignore_index=True)


                users = users.rename(columns={0: 'userid'})
                userswithreviewembs = pd.merge(users, meanuserreviewemb, how="left", left_on="userid",
                                               right_on="user").fillna(0)

                userswithreviewembs = userswithreviewembs.drop(['user'], axis=1)

                userswithreviewembs.to_csv(filepath_write_filterusers_withreviewemb, index=False, header=False)
                # concat clusters
                # ------------------------
                # embeddings of users, items and train file
                users = pd.read_csv(filepath_read_users_concatclusters, header=None)
                items = pd.read_csv(filepath_read_items_concatclusters, index_col=False, header=None)

                userid = users.iloc[:, 0]

                userid_dict = userid.to_dict()
                itemid = items.iloc[:, 0]

                itemid_dict = itemid.to_dict()


                items = items.rename(columns={0: 'id'})


                items['id'] = items['id'].astype(int)
                jsonTrans['id'] = jsonTrans['id'].astype(int)


                itemswithurl = pd.merge(items, jsonTrans, how="left", on="id").fillna(0)

                itemswithurl.to_csv(filepath_write_filtermovies_withreviewurls_concatclusters)


                meanreviewsemb = pd.DataFrame()
                for index, row in itemswithurl.iterrows():
                    # download sentence embeddings folder and run this with e.g.
                    # sent_dir = "/Users/andst/devel/kg_dream_team/data/sentence_embeddings"
                    movieuri = row['URL']
                    if movieuri != 0:
                        embs = load_embeddings(sent_dir, movieuri, split="train")

                    if embs.size > 0:
                        reviews = pd.DataFrame(embs)
                        meanreview = pd.DataFrame(reviews.aggregate('mean')).T
                    else:
                        reviews = pd.DataFrame(np.zeros((1, 768)))
                        meanreview = reviews
                    meanreview['URL'] = movieuri
                    meanreview['id'] = row['id']
                    meanreviewsemb = pd.concat([meanreviewsemb, meanreview], ignore_index=True)


                itemswithreviewembs = pd.merge(itemswithurl, meanreviewsemb, how="left", on="id").fillna(0)

                itemswithreviewembs = itemswithreviewembs.drop(['URL_x', 'URL_y'], axis=1)
                itemswithreviewembs.to_csv(filepath_write_filtermovies_withreviewemb_concatclusters, index=False, header=False)

                meanuserreviewemb = pd.DataFrame()
                for index, row in users.iterrows():
                    userid = row[0]
                    ratedmovies = user_train_ratings.loc[user_train_ratings['user'] == userid]
                    userreviewembeddings = pd.merge(ratedmovies, meanreviewsemb, how="inner", left_on="item", right_on="id")
                    userreviewembeddings = userreviewembeddings.drop(['item', 'URL', 'id', 'rating'], axis=1)
                    meanuserreview = pd.DataFrame(userreviewembeddings.aggregate('mean')).T
                    meanuserreviewemb = pd.concat([meanuserreviewemb, meanuserreview], ignore_index=True)


                users = users.rename(columns={0: 'userid'})
                userswithreviewembs = pd.merge(users, meanuserreviewemb, how="left", left_on="userid",
                                               right_on="user").fillna(0)

                userswithreviewembs = userswithreviewembs.drop(['user'], axis=1)

                userswithreviewembs.to_csv(filepath_write_filterusers_withreviewemb_concatclusters, index=False, header=False)

    elif cfg.reviews:
        for i in range(0, cfg.folds):



            filepath_read_users = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                             + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row.csv"
            filepath_read_items = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                              + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + ".csv"

            filepath_write_filtermovies_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                    "_withconcatreviewemb.csv"
            filepath_write_filterusers_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                    "row_withconcatreviewemb.csv"


            filepath_write_filtermovies_withreviewurls = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                   "_withcontatreviewurl.csv"

            filepath_write_filterusers_withreviewurls = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                    "row_withcontactreviewurl.csv"


            filepath_read_user_train_ratings = cfg.FoldsDir + "test-" + str(i) + ".csv"

            #embeddings of users, items and train file
            users = pd.read_csv(filepath_read_users, header=None)
            user_train_ratings = pd.read_csv(filepath_read_user_train_ratings)
            items = pd.read_csv(filepath_read_items,  index_col=False, header=None)
            print(users)

            #extracting user id and making a dictionary for later use
            userid = users.iloc[:, 0]

            userid_dict = userid.to_dict()

            #extracting itemid and making a dictionary for later use
            itemid = items.iloc[:, 0]

            itemid_dict = itemid.to_dict()


            items = items.rename(columns = {0:'id'})




            items['id'] = items['id'].astype(int)
            jsonTrans['id'] = jsonTrans['id'].astype(int)



            itemswithurl = pd.merge(items, jsonTrans, how="left", on="id").fillna(0)

            itemswithurl.to_csv(filepath_write_filtermovies_withreviewurls)


            meanreviewsemb = pd.DataFrame()
            for index, row in itemswithurl.iterrows():
                # download sentence embeddings folder and run this with e.g.
                movieuri = row['URL']
                if movieuri != 0:
                    embs = load_embeddings(sent_dir, movieuri, split="train")

                if embs.size > 0:
                    reviews = pd.DataFrame(embs)
                    meanreview = pd.DataFrame(reviews.aggregate('mean')).T
                else:
                    reviews = pd.DataFrame(np.zeros((1, 768)))
                    meanreview = reviews
                meanreview['URL'] = movieuri
                meanreview['id'] = row['id']
                meanreviewsemb = pd.concat([meanreviewsemb, meanreview], ignore_index=True)

            itemswithreviewembs = pd.merge(itemswithurl, meanreviewsemb, how="left", on="id").fillna(0)

            itemswithreviewembs = itemswithreviewembs.drop(['URL_x', 'URL_y'], axis=1)
            print("items with reviews:")
            print(itemswithreviewembs)
            itemswithreviewembs.to_csv(filepath_write_filtermovies_withreviewemb, index=False, header=False)

            meanuserreviewemb = pd.DataFrame()
            for index, row in users.iterrows():
                userid = row[0]
                ratedmovies = user_train_ratings.loc[user_train_ratings['user'] == userid]
                userreviewembeddings = pd.merge(ratedmovies, meanreviewsemb, how="inner", left_on="item", right_on="id")
                userreviewembeddings = userreviewembeddings.drop(['item', 'URL', 'id', 'rating'], axis=1)
                meanuserreview = pd.DataFrame(userreviewembeddings.aggregate('mean')).T
                meanuserreviewemb = pd.concat([meanuserreviewemb, meanuserreview], ignore_index=True)

            users = users.rename(columns = {0:'userid'})
            userswithreviewembs = pd.merge(users, meanuserreviewemb, how="left", left_on="userid", right_on="user").fillna(0)

            userswithreviewembs = userswithreviewembs.drop(['user'], axis=1)

            print("users with reviews:")
            print(userswithreviewembs)
            userswithreviewembs.to_csv(filepath_write_filterusers_withreviewemb, index=False, header=False)

    end = dt.datetime.now()
    print("Time of end: " + str(end))
    print("Elapsed time: " + str(end - start))

def run_on_amazon():

    # concat to embeddings

    # load the usersentenceembeddings
    # load the itemssentenceembeddings
    # load userembeddings
    # load itemsembeddings
    # merge the corresponding ones
    # save to appropriate folder
    start = dt.datetime.now()
    print("Time of start: " + str(start))

    if cfg.reviews:
        for i in range(0, cfg.folds):
            filepath_read_users = cfg.SplitToClassDir + "seed" + str(
                cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                                  + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row.csv"
            filepath_read_items = cfg.SplitToClassDir + "seed" + str(
                cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                                  + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + ".csv"


            filepath_read_items_reviewembs = cfg.SentenceEmbDir + "itemreviewsembeddings.csv"


            filepath_read_users_reviewembs = cfg.SentenceEmbDir + "userreviewsembeddings.csv"

            filepath_write_filtermovie_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                        "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                        str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                        "_withconcatreviewemb.csv"
            filepath_write_filterusers_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                       "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                       str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                       "row_withconcatreviewemb.csv"

            users = pd.read_csv(filepath_read_users, index_col=False, header=None)
            items = pd.read_csv(filepath_read_items, index_col=False, header=None)

            items = items.rename(columns={0: 'id'})
            users = users.rename(columns={0: 'id'})


            itemlistfile = "item_list.txt"
            userlistfile = "user_list.txt"
            itemListDF = pd.read_csv(cfg.SentenceEmbDir + itemlistfile, sep=' ')
            userListDF = pd.read_csv(cfg.SentenceEmbDir + userlistfile, sep=' ')

            maxItemID = itemListDF["remap_id"].max()
            maxUserID = userListDF["remap_id"].max()

            offsetForItems = maxUserID + 1
            offsetForEntities = offsetForItems + maxItemID + 1

            items_sentence_emb = pd.read_csv(filepath_read_items_reviewembs, index_col=False, header=None)
            users_sentence_emb = pd.read_csv(filepath_read_users_reviewembs, index_col=False, header=None)

            print(users_sentence_emb)



            print(items_sentence_emb)


            items_sentence_emb = items_sentence_emb.rename(columns={0: 'id'})
            users_sentence_emb = users_sentence_emb.rename(columns={0: 'id'})

            items_sentence_emb["id"] = items_sentence_emb["id"].sub(offsetForItems)

            itemswithreviewembs = pd.merge(items, items_sentence_emb, how="left", left_on="id", right_on="id").fillna(0)

            userswithreviewembs = pd.merge(users, users_sentence_emb, how="left", left_on="id", right_on="id").fillna(0)

            print("items with reviews:")
            print(itemswithreviewembs)
            itemswithreviewembs.to_csv(filepath_write_filtermovie_withreviewemb, index=False, header=False)

            print("users with reviews:")
            print(userswithreviewembs)

            userswithreviewembs.to_csv(filepath_write_filterusers_withreviewemb, index=False, header=False)

    # concat embeddings with clusters

    # load the usersentenceembeddings
    # load the itemssentenceembeddings
    # load userembeddings with clusters
    # load itemsembeddings with clusters
    # merge the corresponding ones
    # save to appropriate folder
    elif cfg.withclustersandreviews:
        for i in range(0, cfg.folds):
            print("Fold: " + str(i))
            for c in range(cfg.minclusters, cfg.maxclusters + 1):
                print("Nr. of clusters: " + str(c))
                meantime = dt.datetime.now()
                print("Current time: " + str(meantime))
                print("Elapsed time so far: " + str(meantime - start))


                # read the original embeddings already with clusters
                # dot clusters

                filepath_read_users = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + "/userEmb_" \
                        + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filtered" \
                        + str(cfg.ratingclass) + "row_" + str(c) + "clusters_withdotclusters.csv"
                filepath_read_items = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + "/movieEmb_" \
                        + cfg.whichratings + "_train" + str(i) + "_seed" + str(cfg.seed) + "filteredclass" \
                        + str(cfg.ratingclass) + "_" + str(c) + "clusters_withdotclusters.csv"

                # concat clusters
                filepath_read_users_concatclusters = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) \
                                                     + "/userEmb_" + cfg.whichratings + "_train" \
                                      + str(i) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row_" \
                                                     + str(c) + "clusters_withconcatclusters.csv"
                filepath_read_items_concatclusters = cfg.EmbeddingsFolderWithClusters + "seed" + str(
                    cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                                      + str(i) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) \
                                                     + "_" + str(c) + "clusters_withconcatclusters.csv"


                # read files with sentence embeddings

                filepath_read_items_reviewembs = cfg.SentenceEmbDir + "itemreviewsembeddings.csv"

                filepath_read_users_reviewembs = cfg.SentenceEmbDir + "userreviewsembeddings.csv"


                # write files
                # with dot clusters and concat reviews
                filepath_write_filtermovies_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                    "_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"
                filepath_write_filterusers_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                    "row_" + str(c) + "clusters_withdotclusters_withconcatreviewemb.csv"

                # with concat clusters and concat reviews

                filepath_write_filtermovies_withreviewemb_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                                                          + str(cfg.seed) + \
                                                            "/movieEmb_" + cfg.whichratings + "_train" + str(i) \
                                                            + "_seed" + str(cfg.seed) + "filteredclass" \
                                                                          + str(cfg.ratingclass) + "_" + str(c) \
                                                            + "clusters_withconcatclusters_withconcatreviewemb.csv"
                filepath_write_filterusers_withreviewemb_concatclusters = cfg.WithReviwesEmbeddingsDir + "seed" \
                                                        + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                                                        + str(i) + "_seed" + str(cfg.seed) + "filtered" \
                                                        + str(cfg.ratingclass) + "row_" + str(c) \
                                                            + "clusters_withconcatclusters_withconcatreviewemb.csv"





                itemlistfile = "item_list.txt"
                userlistfile = "user_list.txt"
                itemListDF = pd.read_csv(cfg.SentenceEmbDir + itemlistfile, sep=' ')
                userListDF = pd.read_csv(cfg.SentenceEmbDir + userlistfile, sep=' ')

                maxItemID = itemListDF["remap_id"].max()
                maxUserID = userListDF["remap_id"].max()

                offsetForItems = maxUserID + 1
                offsetForEntities = offsetForItems + maxItemID + 1

                # dot clusters
                # -----------------
                # embeddings of users, items and train file
                users = pd.read_csv(filepath_read_users, index_col=False, header=None)
                items = pd.read_csv(filepath_read_items, index_col=False, header=None)

                items = items.rename(columns={0: 'id'})
                users = users.rename(columns={0: 'id'})

                print("users:")
                print(users)
                print("items:")
                print(items)



                items_sentence_emb = pd.read_csv(filepath_read_items_reviewembs, index_col=False, header=None)
                users_sentence_emb = pd.read_csv(filepath_read_users_reviewembs, index_col=False, header=None)

                print(items_sentence_emb)

                items_sentence_emb = items_sentence_emb.rename(columns={0: 'id'})
                users_sentence_emb = users_sentence_emb.rename(columns={0: 'id'})

                items_sentence_emb["id"] = items_sentence_emb["id"].sub(offsetForItems)

                itemswithreviewembs = pd.merge(items, items_sentence_emb, how="left", left_on="id",
                                               right_on="id").fillna(0)

                userswithreviewembs = pd.merge(users, users_sentence_emb, how="left", left_on="id",
                                               right_on="id").fillna(0)
                print("items with reviews:")
                print(itemswithreviewembs)
                itemswithreviewembs.to_csv(filepath_write_filtermovies_withreviewemb, index=False, header=False)
                print("users with reviews:")
                print(itemswithreviewembs)
                userswithreviewembs.to_csv(filepath_write_filterusers_withreviewemb, index=False, header=False)

                ################################
                # concat clusters
                ###############################
                users_concutclusters = pd.read_csv(filepath_read_users_concatclusters, index_col=False, header=None)
                items_concutclusters = pd.read_csv(filepath_read_items_concatclusters, index_col=False, header=None)

                items_concutclusters = items.rename(columns={0: 'id'})
                users_concutclusters = users.rename(columns={0: 'id'})
                print(users)
                print(items)


                itemswithreviewembs_concatclusters = pd.merge(items_concutclusters, items_sentence_emb, how="left", left_on="id",
                                               right_on="id").fillna(0)

                userswithreviewembs_concutclusters = pd.merge(users_concutclusters, users_sentence_emb, how="left", left_on="id",
                                               right_on="id").fillna(0)
                print("items with reviews:")
                print(itemswithreviewembs_concatclusters)
                itemswithreviewembs_concatclusters.to_csv(filepath_write_filtermovies_withreviewemb_concatclusters, index=False, header=False)
                print("users with reviews:")
                print(userswithreviewembs_concutclusters)
                userswithreviewembs_concutclusters.to_csv(filepath_write_filterusers_withreviewemb_concatclusters, index=False, header=False)

def get_contatenated_review_embeddings_without_method_embeddings():
    for i in range(0, cfg.folds):
        filepath_read_filtermovies_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                    "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                    str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                    "_withconcatreviewemb.csv"
        filepath_read_filterusers_withreviewemb = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                   "row_withconcatreviewemb.csv"

        filepath_write_filtermovies_withreviewemb_onlyrevies = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                   "/movieEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                   str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                   "_withconcatreviewemb_onlyreviews.csv"
        filepath_write_filterusers_withreviewemb_onlyreviews = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + \
                                                  "/userEmb_" + cfg.whichratings + "_train" + str(i) + "_seed" + \
                                                  str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                  "row_withconcatreviewemb_onlyreviews.csv"
        users = pd.read_csv(filepath_read_filterusers_withreviewemb, header=None)
        items = pd.read_csv(filepath_read_filtermovies_withreviewemb, index_col=False, header=None)
        users = users.drop(users.iloc[:, 1:33], axis=1)
        items = items.drop(items.iloc[:, 1:33], axis=1)
        users.to_csv(filepath_write_filterusers_withreviewemb_onlyreviews, index=False, header=False)
        items.to_csv(filepath_write_filtermovies_withreviewemb_onlyrevies, index=False, header=False)

def main():
    #run_on_mindreader()
    #get_contatenated_review_embeddings_without_method_embeddings
    run_on_amazon()

if __name__ == "__main__":
    main()