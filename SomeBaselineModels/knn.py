#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#not the most optimized KNN recommendation implementation

from sklearn.neighbors import NearestNeighbors
import scipy as scipy
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import config as cfg
import datetime as dt
import random as rnd


def run():
    start = dt.datetime.now()

    print("Time of start: " + str(start))

    for i in range(0, cfg.folds):
        filepath_read_model = cfg.FoldsDir + "train-" + str(i) + ".csv"
        filepath_read_test = cfg.FoldsDir + "test-" + str(i) + ".csv"
        filepath_write = cfg.PredictionResultsDir + "baselines/UserKNN_fold" + str(i) + "_trial.csv"
        movies = pd.read_csv(filepath_read_model)
        movies_test = pd.read_csv(filepath_read_test)

        print("Creating Train Features!")
        #pivot ratings training into movie features
        train_features = movies.pivot(
            index='user',
            columns='item',
            values='rating'
        ).fillna(0)
        # convert dataframe of movie features to scipy sparse matrix
        mat_train_features = csr_matrix(train_features)

        #densemat_movie_features = mat_train_item_features.toarray()

        print("Creating Test Features!")

        # pivot ratings into testing user features
        test_features = movies_test.pivot(
            index='user',
            columns='item',
            values='rating'
        ).fillna(0)
        # convert dataframe of movie features to scipy sparse matrix
        mat_test_features = csr_matrix(test_features)



        model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

        print("Training the model!")
        model_knn.fit(mat_train_features)

        recommendations = pd.DataFrame(columns = ['item', 'score', 'user', 'rank', 'Algorithm'])
        recommendations.to_csv(filepath_write, index=False)

        print("Creating Recommendation List for Recommendation evaluation for KNN!")
        for u in test_features.index:
            #print("user: " + str(u) + ", time: " + str(dt.datetime.now()))
            distances, indices = model_knn.kneighbors(mat_train_features[u], n_neighbors=20)
            # getting sum of distance similarities of neighbors for each test user
            dist = np.array(distances)
            dist = np.delete(dist, [0])
            sumdist = dist.sum()
            recommendationsperuser = pd.DataFrame()

            
            # computing recommendation scores for each user
            for i in test_features.columns:
                #print("item: " + str(i))
                score = 0
                # going through the user neighbors
                for n in indices.flatten():
                    if n != u:
                        r = train_features.loc[n, i]
                        if r != 0:
                            ind = np.where(indices.flatten()==n)[0][0]
                            sim = dist[ind-1]
                            score = score + (sim/sumdist)
                row = {'item': i, 'score': score, 'user': u, 'Algorithm': 'UserKNN'}


                recommendationsperuser = pd.concat([recommendationsperuser, pd.DataFrame([row])], ignore_index=True)


            recommendationsperuser = recommendationsperuser.sort_values(by="score", ascending=False)
            recommendationsperuser.insert(3, "rank", np.arange(1, recommendationsperuser.shape[0] + 1))

            candidatelist = recommendationsperuser.iloc[:cfg.recommendationlistsize]
            #candidatelist = recommendationsperuser
            candidatelist.to_csv(filepath_write, index=False, mode='a', header=False)

            currenttime = dt.datetime.now()

            print("Current user: " + str(u) + ", Current time: " + str(currenttime))


    end = dt.datetime.now()

    print("Time of end: " + str(end))
    print("Duration: " + str(end - start))




def main():
    run()

if __name__ == "__main__":
    main()