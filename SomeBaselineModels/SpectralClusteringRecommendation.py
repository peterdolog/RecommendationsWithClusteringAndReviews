#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#Spectral Clustering based recommendation implementation based on a standard lin alg library
#some helper functions which may be used more broadly
#functions for Amazon applicable for Yelp as well...


import pandas as pd
from pandas.api.types import CategoricalDtype
import sklearn as sk
import scipy as scipy
from scipy.sparse import csr_matrix

import scipy.sparse as spr

import scipy.sparse.linalg as sprlinalg
import config as cfg
import numpy as np
import datetime as dt
import os



def make_train_and_test_for_Neural_Baselines():



    for f in range(0, cfg.folds):
        filepath_read_train = cfg.FoldsDir + "train-" + str(f) + ".csv"
        filepath_read_test = cfg.FoldsDir + "test-" + str(f) + ".csv"
        filepath_write_train = cfg.FoldsDir + "train-KGCL-SIGIR22-main-" + str(f) + ".txt"
        filepath_write_test = cfg.FoldsDir + "test-KGCL-SIGIR22-main-" + str(f) + ".txt"
        trainwritefile = open(filepath_write_train, 'w')
        testwritefile = open(filepath_write_test, 'w')
        traindata = pd.read_csv(filepath_read_train)
        testdata = pd.read_csv(filepath_read_test)

        #positve interactions
        positive = 1
        traindata = traindata[(traindata.rating == positive)]
        testdata = testdata[(testdata.rating == positive)]


        # grouping data based on users and transforming all interactions into one line per each user
        userinteractionstrain = traindata.groupby(['user'])

        for trainname, traingroup in userinteractionstrain:

            trainitemsingroup = list(traingroup['item'])
            trainitemsingroup.insert(0, trainname)
            trainitemsingroup = [str(x) for x in trainitemsingroup]
            for index in range(len(trainitemsingroup) - 1):
                trainitemsingroup[index] = trainitemsingroup[index] + ' '


            trainitemsingroup.append('\n')
            trainwritefile.writelines(trainitemsingroup)



        userinteractionstest = testdata.groupby(['user'])

        for testname, testgroup in userinteractionstest:

            testitemsingroup = list(testgroup['item'])
            testitemsingroup.insert(0, testname)
            testitemsingroup = [str(x) for x in testitemsingroup]
            for index in range(len(testitemsingroup)-1):
                testitemsingroup[index] = testitemsingroup[index] + ' '

            testitemsingroup.append('\n')
            testwritefile.writelines(testitemsingroup)


        trainwritefile.close()
        testwritefile.close()



def make_triples_and_relation_map_for_baselines():
    tripletsfile = cfg.GraphDataDir + "triples.csv"
    mapfile = cfg.GraphDataDir + "maps.csv"
    tripleswithidswritefile = cfg.GraphDataDir + "tripleswithidswritefile.csv"
    relationmapfile = cfg.GraphDataDir + "relationmap.csv"

    triplesdf = pd.read_csv(tripletsfile, index_col=0)
    mapdf = pd.read_csv(mapfile, names=('url', 'id'), sep=' ')

    maxentityid = mapdf['id'].max() + 1
    print(maxentityid)



    tripleswithidshead = pd.merge(triplesdf, mapdf, how="inner", left_on="head_uri", right_on="url")
    tripleswithidshead.rename(columns={'id': 'head'}, inplace=True)



    tripleswithids = pd.merge(tripleswithidshead, mapdf, how="inner", left_on="tail_uri", right_on="url")
    tripleswithids.rename(columns={'id': 'tail'}, inplace=True)



    tripleswithids = tripleswithids.drop(columns=['head_uri', 'tail_uri', 'url_x', 'url_y'])

    ltripleswithids = tripleswithids[['head', 'relation', 'tail']]



    #mapping relations names to ids
    #relations = pd.DataFrame()
    relations = ltripleswithids['relation']
    relations = relations.drop_duplicates()
    relations = relations.reset_index()
    relations = relations.drop(columns=['index'])
    relations = relations.reset_index()
    relations.rename(columns={'index': 'relationid'}, inplace=True)

    print(relations)
    #replacing relations names with ids based on relation map above
    tripleswithrelids = pd.merge(ltripleswithids, relations, how="inner", left_on="relation", right_on="relation")
    tripleswithrelids = tripleswithrelids.drop(columns=['relation'])
    tripleswithrelids = tripleswithrelids[['head', 'relationid', 'tail']]
    print(tripleswithrelids)

    tripleswithrelids.to_csv(tripleswithidswritefile, sep=' ', header=None, index=False)
    relations.to_csv(relationmapfile, header=None, index=False)


def make_triples_and_relation_map_for_baselines_items_first():

    tripletsfile = cfg.GraphDataDir + "triples.csv"
    mapfile = cfg.GraphDataDir + "maps.csv"
    entitiesfile = cfg.GraphDataDir + "entities.csv"
    tripleswithidswritefile = cfg.GraphDataDir + "tripleswithidswritefile.csv"
    relationmapfile = cfg.GraphDataDir + "relationmap.csv"

    triplesdf = pd.read_csv(tripletsfile, index_col=0)
    mapdf = pd.read_csv(mapfile, names=('url', 'id'), sep=' ')
    entitiesdf = pd.read_csv(entitiesfile)

    movieentitiesdf = entitiesdf[(entitiesdf.labels == 'Movie')]



    triples_with_itemsonly_in_first_column = pd.merge(triplesdf, movieentitiesdf, how="inner", left_on= "head_uri",
                                             right_on="uri")





    triples_with_itemsonly_in_tail_column = pd.merge(triples_with_itemsonly_in_first_column,
                                                     movieentitiesdf,
                                                     how="inner",
                                                     left_on="tail_uri",
                                                     right_on="uri")

    triples_with_itemsonly_in_first_column = triples_with_itemsonly_in_first_column[['head_uri', 'relation', 'tail_uri']]


    maxentityid = mapdf['id'].max() + 1
    print(maxentityid)


    tripleswithidshead = pd.merge(triples_with_itemsonly_in_first_column, mapdf, how="inner", left_on="head_uri", right_on="url")
    tripleswithidshead.rename(columns={'id': 'head'}, inplace=True)


    tripleswithids = pd.merge(tripleswithidshead, mapdf, how="inner", left_on="tail_uri", right_on="url")
    tripleswithids.rename(columns={'id': 'tail'}, inplace=True)


    tripleswithids = tripleswithids.drop(columns=['head_uri', 'tail_uri', 'url_x', 'url_y'])

    ltripleswithids = tripleswithids[['head', 'relation', 'tail']]


    #mapping relations names to ids
    relations = ltripleswithids['relation']
    relations = relations.drop_duplicates()
    relations = relations.reset_index()
    relations = relations.drop(columns=['index'])
    relations = relations.reset_index()
    relations.rename(columns={'index': 'relationid'}, inplace=True)

    print(relations)
    #replacing relations names with ids based on relation map above
    tripleswithrelids = pd.merge(ltripleswithids, relations, how="inner", left_on="relation", right_on="relation")
    tripleswithrelids = tripleswithrelids.drop(columns=['relation'])
    tripleswithrelids = tripleswithrelids[['head', 'relationid', 'tail']]
    print(tripleswithrelids)

    tripleswithrelids.to_csv(tripleswithidswritefile, sep=' ', header=None, index=False)
    relations.to_csv(relationmapfile, header=None, index=False)

def create_eigenvectors_MindReader():
    start = dt.datetime.now()
    print("Time of start of creating eigen values and eigen vectors: " + str(start))
    tripletsfile = cfg.GraphDataDir + "triples.csv"
    mapfile = cfg.GraphDataDir + "maps.csv"




    triplesdf = pd.read_csv(tripletsfile, index_col=0)
    mapdf = pd.read_csv(mapfile, names=('url', 'id'), sep=' ')

    maxentityid = mapdf['id'].max() + 1
    print(maxentityid)


    tripleswithidshead = pd.merge(triplesdf, mapdf, how="inner", left_on="head_uri", right_on="url")
    tripleswithidshead.rename(columns = {'id':'head'}, inplace = True)


    tripleswithids = pd.merge(tripleswithidshead, mapdf, how="inner", left_on="tail_uri", right_on="url")
    tripleswithids.rename(columns = {'id':'tail'}, inplace = True)


    tripleswithids = tripleswithids.drop(columns=['head_uri', 'tail_uri', 'url_x', 'url_y'])

    tripleswithids = tripleswithids[['head', 'relation', 'tail']]



    tripleswithids_incident = tripleswithids
    tripleswithids_incident['relation'] = 1


    tripleswithids_incident.drop_duplicates(inplace=True)

    for f in range(0, cfg.folds):
        timestartfold = dt.datetime.now()
        print("Starting fold " + str(f) + " at " + str(timestartfold))
        filepath_read_train = cfg.FoldsDir + "train-" + str(f) + ".csv"
        eigenvalwritefile = cfg.GraphDataDir + "eigenvalues-" + str(f) + ".csv"
        eigenvecswritefile = cfg.GraphDataDir + "eigenvectors-" + str(f) + ".csv"
        usersmapfile = cfg.GraphDataDir + "usersmap" + str(f) + ".csv"

        traindata = pd.read_csv(filepath_read_train)

        tripleswithidsfoldhead = pd.merge(tripleswithids_incident, traindata, how="inner", left_on="head", right_on="item")

        tripleswithidsfoldhead = tripleswithidsfoldhead.drop(columns=['user', 'rating', 'item'])
        tripleswithidsfoldhead.drop_duplicates(inplace=True)
        tripleswithidsfoldtail = pd.merge(tripleswithids_incident, traindata, how="inner", left_on="tail",
                                          right_on="item")
        tripleswithidsfoldtail = tripleswithidsfoldtail.drop(columns=['user', 'rating', 'item'])
        tripleswithidsfoldtail.drop_duplicates(inplace=True)
        tripleswithidsfold = pd.concat([tripleswithidsfoldhead, tripleswithidsfoldtail], ignore_index=True)
        tripleswithidsfold.drop_duplicates(inplace=True)

        traindata= traindata.drop(columns=['rating'])
        users = traindata['user']
        items = traindata['item']

        users.drop_duplicates(inplace=True)
        items.drop_duplicates(inplace=True)


        items = items.reset_index()
        users = users.reset_index()
        items = items.drop(columns=['index'])
        users = users.drop(columns=['index'])



        users.insert(1, "newid", np.arange(maxentityid, users.shape[0] + maxentityid))
        users.to_csv(usersmapfile, index=False)
        traindatanewuserid = pd.merge(traindata, users, how="inner", left_on="user", right_on="user" )


        traindatanewuserid['relation'] = 1
        traindatanewuserid = traindatanewuserid.drop(columns=['user'])

        traindatanewuserid.rename(columns={'item': 'tail', 'newid':'head'}, inplace=True)
        traindatanewuserid = traindatanewuserid[['head', 'relation', 'tail']]
        traindatanewuserid.drop_duplicates(ignore_index=True)

        tripleswithidsfoldwithusers = pd.concat([tripleswithidsfold, traindatanewuserid], ignore_index=True)

        # getting a matrix of graph
        A = tripleswithidsfoldwithusers.pivot(
            index='head',
            columns='tail',
            values='relation'
        ).fillna(0)

        # reshaping to squared matrix in order to be able to do PCA
        indexA = A.index.union(A.columns)
        A = A.reindex(index=indexA, columns=indexA, fill_value=0)
        # computing diagonal matrix
        D = np.diag(A.sum(axis=1))


        L = D - A


        eigenval, eigenvecs = np.linalg.eig(L)




        sorted_eigen_values = np.argsort(eigenval)[::-1]
        sorted_eigenvectors = eigenvecs[:, sorted_eigen_values]

        selected_eigen_vectors = np.real(sorted_eigenvectors[:32])


        eigenvaldf = pd.DataFrame(sorted_eigen_values[:32])
        eigenvecsdf = pd.DataFrame(selected_eigen_vectors)

        eigenvaldf.to_csv(eigenvalwritefile, header=None, index=False)
        eigenvecsdf.to_csv(eigenvecswritefile, header=None, index=False)
        timeendfold = dt.datetime.now()
        print("finished fold " + str(f) + " at " + str(timeendfold) + " with duration " + str(timeendfold - timestartfold))

    end = dt.datetime.now()
    print("Finished creation of eigen values and eigen vectors: " + str(end))
    print("Duration: " + str(end - start))

def create_triples_out_of_trainInteractions():
    interactionFile = cfg.GraphDataDir + "AmazonBook/train.txt"
    #interactionFile = cfg.GraphDataDir + "Yelp2018/train.txt"
    #interactionFile = cfg.GraphDataDir + "AmazonBook/trainExample.txt"
    interactionTriples = cfg.GraphDataDir + "AmazonBook/interactionTriples.csv"
    #interactionTriples = cfg.GraphDataDir + "Yelp2018/interactionTriples.csv"
    interactionFileHandler = open(interactionFile, mode='r')
    userInteractionLines = interactionFileHandler.readlines()
    interactionFileHandler.close()

    trainUniqueUsers, trainRelation, trainItem, trainUser = [], [], [], []

    for line in userInteractionLines:
        if len(line) > 0:
            l = line.strip('\n').split(' ')
            if l[1]:
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainRelation.extend([39] * len(items))
                trainItem.extend(items)

    print('Users: ' + str(len(trainUser)) +
          ', Relations: ' + str(len(trainRelation)) +
          ', Items: ' +  str(len(trainUser))
                                )

    R = {'head': trainUser, 'relation': trainRelation, 'tail': trainItem}

    interactionDF = pd.DataFrame(R)


    print(interactionDF)
    interactionDF.to_csv(interactionTriples, index=False)



def create_eigenvectors_AmazonBook():
    start = dt.datetime.now()
    print("Time of start of creating eigen values and eigen vectors: " + str(start))



    tripletsfile = cfg.GraphDataDir + "AmazonBook/kg_final.txt"

    interactionTriples = cfg.GraphDataDir + "AmazonBook/interactionTriples.csv"

    itemlistfile = cfg.GraphDataDir + "AmazonBook/item_list.txt"
    userlistfile = cfg.GraphDataDir + "AmazonBook/user_list.txt"


    eigenvalwritefile = cfg.GraphDataDir + "eigenvalues-AmazonBook.csv"
    eigenvecswritefile = cfg.GraphDataDir + "eigenvectors-AmazonBook.csv"

    itemListDF = pd.read_csv(itemlistfile, sep=' ')
    userListDF = pd.read_csv(userlistfile, sep=' ')



    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()


    offsetForItems = maxUserID + 1
    offsetForEntities = offsetForItems + maxItemID + 1

    if os.path.exists(interactionTriples):
        interactionTriplesDF = pd.read_csv(interactionTriples)
    else:
        print("Creating graph out of interactions ...")
        create_triples_out_of_trainInteractions()
        interactionTriplesDF = pd.read_csv(interactionTriples)


    kgtriplesDF = pd.read_csv(tripletsfile, header=None, names=('head', 'relation', 'tail'), sep=' ')


    kgtriplesDF["tail"] = kgtriplesDF["tail"].add(offsetForEntities)
    kgtriplesDF["relation"] = 1

    interactionTriplesDF["tail"] = interactionTriplesDF["tail"].add(offsetForItems)
    interactionTriplesDF["relation"] = 1

    interactionKGDF = pd.concat([interactionTriplesDF, kgtriplesDF], ignore_index=True)

    print(interactionTriplesDF)
    print(kgtriplesDF)
    print(interactionKGDF)

    interactionKGDF.drop_duplicates(inplace=True)

    interactionKGDF = interactionKGDF.reset_index()



    print("Performing Spectral Clustering ...")

    A = interactionKGDF.pivot(
            index='head',
            columns='tail',
            values='relation'
        ).fillna(0)

    # reshaping to squared matrix in order to be able to do PCA
    indexA = A.index.union(A.columns)
    A = A.reindex(index=indexA, columns=indexA, fill_value=0)
    print(A)
    # computing diagonal matrix
    D = np.diag(A.sum(axis=1))

    print(D)

    L = D - A

    print(L)


    eigenval, eigenvecs = np.linalg.eig(L)




    sorted_eigen_values = np.argsort(eigenval)[::-1]
    sorted_eigenvectors = eigenvecs[:, sorted_eigen_values]

    selected_eigen_vectors = np.real(sorted_eigenvectors[:32])


    eigenvaldf = pd.DataFrame(sorted_eigen_values[:32])
    eigenvecsdf = pd.DataFrame(selected_eigen_vectors)

    eigenvaldf.to_csv(eigenvalwritefile, header=None, index=False)
    eigenvecsdf.to_csv(eigenvecswritefile, header=None, index=False)

    end = dt.datetime.now()
    print("Finished creation of eigen values and eigen vectors: " + str(end))
    print("Duration: " + str(end - start))


def create_eigenvectors_AmazonBook_sparse():
    start = dt.datetime.now()
    print("Time of start of creating eigen values and eigen vectors: " + str(start))

    tripletsfile = cfg.GraphDataDir + "AmazonBook/kg_final.txt"

    interactionTriples = cfg.GraphDataDir + "AmazonBook/interactionTriples.csv"

    itemlistfile = cfg.GraphDataDir + "AmazonBook/item_list.txt"
    userlistfile = cfg.GraphDataDir + "AmazonBook/user_list.txt"


    eigenvalwritefile = cfg.GraphDataDir + "eigenvalues-AmazonBook-corrected.csv"
    eigenvecswritefile = cfg.GraphDataDir + "eigenvectors-AmazonBook-corrected.csv"


    itemListDF = pd.read_csv(itemlistfile, sep=' ')
    userListDF = pd.read_csv(userlistfile, sep=' ')



    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()


    offsetForItems = maxUserID + 1
    offsetForEntities = offsetForItems + maxItemID + 1

    if os.path.exists(interactionTriples):
        interactionTriplesDF = pd.read_csv(interactionTriples)
    else:
        print("Creating graph out of interactions ...")
        create_triples_out_of_trainInteractions()
        interactionTriplesDF = pd.read_csv(interactionTriples)


    kgtriplesDF = pd.read_csv(tripletsfile, header=None, names=('head', 'relation', 'tail'), sep=' ')


    kgtriplesDF["tail"] = kgtriplesDF["tail"].add(offsetForEntities)
    kgtriplesDF["relation"] = 1

    interactionTriplesDF["tail"] = interactionTriplesDF["tail"].add(offsetForItems)
    interactionTriplesDF["relation"] = 1

    interactionKGDF = pd.concat([interactionTriplesDF, kgtriplesDF], ignore_index=True)


    interactionKGDF.drop_duplicates(inplace=True)

    interactionKGDF = interactionKGDF.reset_index()





    head_dim = interactionKGDF["head"]
    tail_dim = interactionKGDF["tail"]




    shape = (len(head_dim), len(tail_dim))


    # Conversion via COO matrix
    coo = scipy.sparse.coo_matrix((interactionKGDF["relation"], (head_dim, tail_dim)), shape=shape)
    A = coo.tocsr()
    A = A.asfptype()



    D = spr.diags(A.sum(axis=1).A1, shape=shape)



    L = D - A


    print("Performing Spectral Clustering ...")

    #since the SM method takes too long, we use shift-invert mode
    # - egenvalues cloes to 0 - specified in sigma - and then LM
    # eigenval, eigenvecs = sprlinalg.eigs(L, k=32, sigma=0)

    # smallest
    eigenval, eigenvecs = sprlinalg.eigs(L, k=32, which='SM')


    print(len(eigenval))
    print(eigenval)




    selected_eigen_vectors = np.real(eigenvecs)


    print(selected_eigen_vectors)


    eigenvaldf = pd.DataFrame(eigenval)
    eigenvecsdf = pd.DataFrame(selected_eigen_vectors)

    eigenvaldf.to_csv(eigenvalwritefile, header=None, index=False)
    eigenvecsdf.to_csv(eigenvecswritefile, header=None, index=False)

    end = dt.datetime.now()
    print("Finished creation of eigen values and eigen vectors: " + str(end))
    print("Duration: " + str(end - start))



def make_embeddings_MindReader():
    start = dt.datetime.now()
    print("Start of creating embeddings for users and items: " + str(start))

    tripletsfile = cfg.GraphDataDir + "triples.csv"
    mapfile = cfg.GraphDataDir + "maps.csv"

    triplesdf = pd.read_csv(tripletsfile, index_col=0)
    mapdf = pd.read_csv(mapfile, names=('url', 'id'), sep=' ')
    maxentityid = mapdf['id'].max() + 1

    for f in range(0, cfg.folds):
        filepath_read_train = cfg.FoldsDir + "train-" + str(f) + ".csv"
        eigenvalreadfile = cfg.GraphDataDir + "eigenvalues-" + str(f) + ".csv"
        eigenvecsreadfile = cfg.GraphDataDir + "eigenvectors-" + str(f) + ".csv"
        usersmapfile = cfg.GraphDataDir + "usersmap" + str(f) + ".csv"
        userembwritefile = cfg.SpectralClusteringEmbeddingDir + "userEmb_allRatings_train" + str(f) + ".csv"
        itemembwritefile = cfg.SpectralClusteringEmbeddingDir + "movieEmb_allRatings_train" + str(f) + ".csv"
        eigenvectors = pd.read_csv(eigenvecsreadfile,header=None, index_col=False)
        usersmap = pd.read_csv(usersmapfile)
        eigenvectorsT = eigenvectors.T

        traindata = pd.read_csv(filepath_read_train)

        usersemb = eigenvectorsT[eigenvectorsT.index >= maxentityid]
        itemsemb = eigenvectorsT[eigenvectorsT.index < maxentityid]
        print(itemsemb)

        usersemb.reset_index(inplace=True)
        itemsemb.reset_index(inplace=True)

        usersemb = pd.merge(usersemb, usersmap, how="inner", left_on="index", right_on="newid")
        usersemb = usersemb.drop(columns=['index', 'newid'])

        column_to_move = usersemb.pop("user")
        usersemb.insert(0, "user", column_to_move)
        usersemb.to_csv(userembwritefile, header=None, index=False)

        itemsemb = pd.merge(itemsemb, traindata, how="inner", left_on="index", right_on="item")
        itemsemb = itemsemb.drop(columns=['index', 'user', 'rating'])
        itemsemb = itemsemb.drop_duplicates(ignore_index=True)
        column_to_move = itemsemb.pop("item")
        itemsemb.insert(0, "item", column_to_move)
        itemsemb.to_csv(itemembwritefile, header=None, index=False)
    end = dt.datetime.now()
    print("Finished creation of embeddings of items and users: " + str(end))
    print("Duration: " + str(end - start))


def make_embeddings_AmazonBook():
    start = dt.datetime.now()
    print("Start of creating embeddings for users and items: " + str(start))
    itemlistfile = cfg.GraphDataDir + "AmazonBook/item_list.txt"
    userlistfile = cfg.GraphDataDir + "AmazonBook/user_list.txt"
    #itemlistfile = cfg.GraphDataDir + "Yelp2018/item_list.txt"
    #userlistfile = cfg.GraphDataDir + "Yelp2018/user_list.txt"


    eigenvecsreadfile = cfg.GraphDataDir + "eigenvectors-AmazonBook-corrected.csv"
    #eigenvecsreadfile = cfg.GraphDataDir + "eigenvectors-Yelp2018.csv"

    itemListDF = pd.read_csv(itemlistfile, sep=' ')
    userListDF = pd.read_csv(userlistfile, sep=' ')

    maxItemID = itemListDF["remap_id"].max()
    maxUserID = userListDF["remap_id"].max()

    for f in range(0, cfg.folds):


        userembwritefile = cfg.GraphDataDir + "AmazonCorrected_userEmb_allRatings_train" + str(f) + ".csv"
        itemembwritefile = cfg.GraphDataDir + "AmazonCorrected_movieEmb_allRatings_train" + str(f) + ".csv"



        eigenvectors = pd.read_csv(eigenvecsreadfile,header=None, index_col=False)


        offsetForItems = maxUserID + 1
        offsetForEntities = offsetForItems + maxItemID + 1
        usersemb = eigenvectors[eigenvectors.index <= maxUserID]
        itemandentsemb = eigenvectors[eigenvectors.index >= offsetForItems]
        itemsemb = itemandentsemb[itemandentsemb.index < offsetForEntities]



        usersemb.reset_index(inplace=True)
        itemsemb.reset_index(inplace=True)

        itemsemb['index'] = itemsemb['index'] - offsetForItems

        print(usersemb)
        print(itemsemb)
        usersemb.to_csv(userembwritefile, header=None, index=False)
        #print(usersemb)

        itemsemb.to_csv(itemembwritefile, header=None, index=False)
    end = dt.datetime.now()
    print("Finished creation of embeddings of items and users: " + str(end))
    print("Duration: " + str(end - start))

def main():
    #make_triples_and_relation_map_for_baselines()
    #make_train_and_test_for_Neural_Baselines()
    #make_triples_and_relation_map_for_baselines_items_first()

    #create_triples_out_of_trainInteractions()
    #create_eigenvectors_AmazonBook_sparse()
    make_embeddings_AmazonBook()



if __name__ == "__main__":
    main()