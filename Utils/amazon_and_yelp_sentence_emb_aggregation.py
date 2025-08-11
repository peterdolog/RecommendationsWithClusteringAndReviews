#Peter Dolog, Aalborg University, dolog@cs.aau.dk
# Aggregates review embeddings per users and items for amazon and yelp datasets
# contains additional helper functions

import pandas as pd
import joblib as jb
import numpy as np
import json as js
import datetime as dt

from typing import List, Union
import os

from fontTools.subset import remap

import config as cfg

import gzip

# load data using Python JSON module
def parseJson(path):
    with open(path, 'r') as f:
        data = f.readlines()
        data = [js.loads(d) for d in data]
    # Flatten data
    df_nested_list = pd.json_normalize(data)
    #print(df_nested_list)
    return df_nested_list





#load embeddings of reviews for URLs
def load_embeddings(sentence_emb_dir: str, movie_uris: Union[str, List[str]], split="train"):
    if isinstance(movie_uris, str):
        movie_uris = [movie_uris]

    df = pd.read_csv(os.path.join(sentence_emb_dir, f"{split}_df_with_uris.csv"), sep=";")
    sentence_embs = jb.load(os.path.join(sentence_emb_dir, f"{split}_embeddings.pbz2"))

    id_df = df[df["ml_uri"].isin(movie_uris)]

    embs = sentence_embs[id_df["id"].values, :]
    return embs


# loads the whole embeddings from a dump file
def load_the_whole_embeddings(sentence_emb_dir: str, split="test"):
    sentence_embs = jb.load(os.path.join(sentence_emb_dir, f"{split}_embeddings.pbz2"))

    #print(sentence_embs)

    sentence_embs_DF = pd.DataFrame(sentence_embs)


    return sentence_embs_DF


def create_user_and_item_embedings(sentence_emb):
    itemlistfile = cfg.GraphDataDir + "AmazonBook/item_list.txt"
    userlistfile = cfg.GraphDataDir + "AmazonBook/user_list.txt"
    itemListDF = pd.read_csv(itemlistfile, sep=' ')
    userListDF = pd.read_csv(userlistfile, sep=' ')

    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()

    offsetForItems = maxUserID
    offsetForEntities = maxUserID + maxItemID

    itemsreviews = pd.merge(sentence_emb, itemListDF, left_on="id", right_on="asin").fillna(0)
    userreviews = pd.merge(sentence_emb, userListDF, left_on="id", right_on="reviewerID").fillna(0)



# merging sentence embeddings with review file to get proper ids for dereferencing in prediction
# using of join because I merge on index. Each embedding row corresponds to a row in the reviews file
def getMergedEmbeddings(review_text_df, sentence_embeddings):
    return sentence_embeddings.join(review_text_df)

def create_triples_out_of_Interactions():
    #interactionFile = cfg.GraphDataDir + "AmazonBook/test.txt"
    interactionFile = cfg.GraphDataDir + "test.txt"
    #interactionFile = cfg.GraphDataDir + "AmazonBook/train.txt"
    #interactionFile = cfg.GraphDataDir + "train.txt"
    #interactionFile = cfg.GraphDataDir + "AmazonBook/trainExample.txt"
    #interactionTriples = cfg.GraphDataDir + "AmazonBook/trainTriples.csv"
    #interactionTriples = cfg.GraphDataDir + "trainTriples.csv"
    interactionTriples = cfg.GraphDataDir + "testTriples.csv"
    interactionFileHandler = open(interactionFile, mode='r')
    userInteractionLines = interactionFileHandler.readlines()
    interactionFileHandler.close()

    testUniqueUsers, testRelation, testItem, testUser = [], [], [], []


    for line in userInteractionLines:
        if len(line) > 0:
            l = line.strip('\n').split(' ')
            #print(l)
            if l[1]:
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testRelation.extend([1] * len(items))
                testItem.extend(items)

    print('Users: ' + str(len(testUser)) +
          ', Relations: ' + str(len(testRelation)) +
          ', Items: ' +  str(len(testUser))
                                )

    R = {'user': testUser, 'item': testItem, 'rating': testRelation}

    interactionDF = pd.DataFrame(R)



    print(interactionDF)
    interactionDF.to_csv(interactionTriples, index=False)


def create_triples_out_of_Interactions_Yelp():
    #interactionFile = cfg.GraphDataDir + "Yelp/test.txt"
    #interactionFile = cfg.GraphDataDir + "Yelp/train.txt"
    #interactionFile = cfg.GraphDataDir + "Yelp/trainExample.txt"
    #interactionFile = "Data/folds/yelpfolds/train.txt"
    interactionFile = "Data/folds/yelpfolds/test.txt"

    #interactionTriples = cfg.GraphDataDir + "Yelp/trainTriples.csv"
    #interactionTriples = "Data/folds/yelpfolds/trainTriples.csv"
    interactionTriples = "Data/folds/yelpfolds/testTriples.csv"

    interactionFileHandler = open(interactionFile, mode='r')
    userInteractionLines = interactionFileHandler.readlines()
    interactionFileHandler.close()

    testUniqueUsers, testRelation, testItem, testUser = [], [], [], []

    for line in userInteractionLines:
        if len(line) > 0:
            l = line.strip('\n').split(' ')
            #print(l)
            if l[1]:
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testRelation.extend([1] * len(items))
                testItem.extend(items)

    print('Users: ' + str(len(testUser)) +
          ', Relations: ' + str(len(testRelation)) +
          ', Items: ' +  str(len(testUser))
                                )

    R = {'user': testUser, 'item': testItem, 'rating': testRelation}

    interactionDF = pd.DataFrame(R)
   

    print(interactionDF)
    interactionDF.to_csv(interactionTriples, index=False)


def merge_amazon():
    sent_dir = "Data/amazonreviewembeddings/"
    file = "sub_books_Ab.txt"
    path = sent_dir + file

    review_text_df = parseJson(path)




    sentence_embeddings = load_the_whole_embeddings(sent_dir)



    merged_embeddings = getMergedEmbeddings(review_text_df, sentence_embeddings)



    itemlistfile = "item_list.txt"
    userlistfile = "user_list.txt"
    itemListDF = pd.read_csv(sent_dir + itemlistfile, sep=' ')
    userListDF = pd.read_csv(sent_dir + userlistfile, sep=' ')

    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()

    offsetForItems = maxUserID + 1
    offsetForEntities = offsetForItems + maxItemID + 1

    userreviewsembsFile = "userreviewsembeddings.csv"

    itemreviewsembsFile = "itemreviewsembeddings.csv"



    # merging sentence review embeddings with users
    userreviewembedings = pd.merge(merged_embeddings, userListDF, how="inner", left_on="reviewerID", right_on="org_id")

    # merging sentence review embeddings with items
    itemsreviewsembeddings = pd.merge(merged_embeddings, itemListDF, how="inner", left_on="asin", right_on="org_id")


    # I only need the remap_id for inference, the other columns can be dropped
    userreviewembedings = userreviewembedings.drop(columns=['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style.Format:', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'image',   'org_id'])

    # I only need the remap_id for inference, the other columns can be dropped
    itemsreviewsembeddings = itemsreviewsembeddings.drop(columns=['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style.Format:', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'image',   'org_id', 'freebase_id'])


    print(userreviewembedings)

    # aggregating sentence embeddings per user to get one mean average for each user that they can be concatenated
    userreviewembedings_aggregated = userreviewembedings.groupby('remap_id').aggregate('mean').reset_index()

    print(userreviewembedings_aggregated)


    userreviewembedings_aggregated.to_csv(sent_dir + userreviewsembsFile, index=False, header=False)

    print(itemsreviewsembeddings)

    # aggregating sentence embeddings per user to get one mean average for each item that they can be concatenated
    itemsreviewsembeddings_aggregated = itemsreviewsembeddings.groupby('remap_id').aggregate('mean').reset_index()

    # moving ids up that they can be identifiable in the inference in the same way as they are in user embeddings

    itemsreviewsembeddings_aggregated["remap_id"] = itemsreviewsembeddings_aggregated["remap_id"].add(offsetForItems)

    print(itemsreviewsembeddings_aggregated)

    itemsreviewsembeddings_aggregated.to_csv(sent_dir + itemreviewsembsFile, index=False, header=False)


def merge_yelp():
    sent_dir = "Data/yelpreviewembeddings/KGATreviewemb/"
    reviewsfile = "filtered_reviews_KGAT.csv"
    embeddingsfile = "text_embeddings_KGAT.pkl.gz"
    reviewspath = sent_dir + reviewsfile
    embeddingspath = sent_dir + embeddingsfile


    review_text_df = pd.read_csv(reviewspath)





    sentence_embeddings_np = jb.load(embeddingspath)

    sentence_embeddings = pd.DataFrame(sentence_embeddings_np)




    merged_embeddings = getMergedEmbeddings(review_text_df, sentence_embeddings)
    print(merged_embeddings['business_id'])


    itemlistfile = "item_list.txt"
    userlistfile = "user_list.txt"
    itemListDF = pd.read_csv(sent_dir + itemlistfile, sep=' ')
    userListDF = pd.read_csv(sent_dir + userlistfile, sep=' ')



    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()

    offsetForItems = maxUserID + 1
    offsetForEntities = offsetForItems + maxItemID + 1

    userreviewsembsFile = "yelp_userreviewsembeddings.csv"

    itemreviewsembsFile = "yelp_itemreviewsembeddings.csv"


    print(itemListDF)
    #print(userListDF)

    # merging sentence review embeddings with users
    userreviewembedings = pd.merge(merged_embeddings, userListDF, how="inner", left_on="user_id", right_on="org_id")

    # merging sentence review embeddings with items
    itemsreviewsembeddings = pd.merge(merged_embeddings, itemListDF, how="inner", left_on="business_id", right_on="org_id")

    print(itemsreviewsembeddings)

    # I only need the remap_id for inference, the other columns can be dropped
    userreviewembedings = userreviewembedings.drop(columns=['text', 'review_id', 'user_id', 'business_id', 'stars', 'date', 'org_id'])

    # I only need the remap_id for inference, the other columns can be dropped
    itemsreviewsembeddings = itemsreviewsembeddings.drop(columns=['text', 'review_id', 'user_id', 'business_id', 'stars', 'date', 'org_id'])


    print(userreviewembedings)

    # aggregating sentence embeddings per user to get one mean average for each user that they can be concatenated
    userreviewembedings_aggregated = userreviewembedings.groupby('remap_id').aggregate('mean').reset_index()

    print(userreviewembedings_aggregated)


    userreviewembedings_aggregated.to_csv(sent_dir + userreviewsembsFile, index=False, header=False)

    print(itemsreviewsembeddings)

    # aggregating sentence embeddings per user to get one mean average for each item that they can be concatenated
    itemsreviewsembeddings_aggregated = itemsreviewsembeddings.groupby('remap_id').aggregate('mean').reset_index()

    # moving ids up that they can be identifiable in the inference in the same way as they are in user embeddings

    itemsreviewsembeddings_aggregated["remap_id"] = itemsreviewsembeddings_aggregated["remap_id"].add(offsetForItems)

    print(itemsreviewsembeddings_aggregated)

    itemsreviewsembeddings_aggregated.to_csv(sent_dir + itemreviewsembsFile, index=False, header=False)


def correcting_dataset():
    sent_dir = "Data/yelpreviewembeddings/KGATreviewemb/"
    itemlistfile = "item_list.txt"
    itemlistfile_corrected = "item_list_crr.txt"
    itemListDF = pd.read_csv(sent_dir + itemlistfile, sep=' ')
    print(itemListDF)

    itemListDF = itemListDF.drop('freebase_id', axis=1)
    itemListDF = itemListDF.drop('remap_id', axis=1)
    print(itemListDF)

    # itemListDF['freebase_id'] = itemListDF['remap_id'].str.split(n=1,expand=True)
    itemListDF['remap_id'] = itemListDF.index

    print(itemListDF)



    itemListDF.to_csv(sent_dir + itemlistfile_corrected, index=False, sep=' ')




def main():
    #create_triples_out_of_Interactions()
    create_triples_out_of_Interactions_Yelp()
    #merge_yelp()
    #correcting_dataset()

if __name__ == "__main__":
    main()