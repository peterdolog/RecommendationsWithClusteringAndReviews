#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#ALS and BPR implementation based on implicit library. BPR implementation from LightGCN implementation performs better. This is why only ALS is used from this library.


import implicit
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
import config as cfg
import datetime as dt

def run():
    for i in range(0, cfg.folds):
        filepath_read_model = cfg.FoldsDir + "train-" + str(i) + ".csv"
        filepath_read_test = cfg.FoldsDir + "test-" + str(i) + ".csv"
        filepath_write = cfg.PredictionResultsDir + "baselines/implMF_fold" + str(i) + ".csv"

        movies = pd.read_csv(filepath_read_model)
        movies_test = pd.read_csv(filepath_read_test)


        start = dt.datetime.now()
        print("Time of start: " + str(start))

        print("Creating Train Features!")
        train_user_features = movies.pivot(
            index='user',
            columns='item',
            values='rating'
        ).fillna(0)
        # convert dataframe of movie features to scipy sparse matrix
        mat_train_user_features = csr_matrix(train_user_features.values)


        print("Creating Test Features!")
        test_file_features = movies_test.pivot(
            index='user',
            columns='item',
            values='rating'
        ).fillna(0)

        # convert dataframe of movie features to scipy sparse matrix
        mat_test_file_features = csr_matrix(test_file_features.values)

        model = implicit.als.AlternatingLeastSquares(factors=150)
        model2 = implicit.bpr.BayesianPersonalizedRanking(factors=150)

        print("Training ALS!")
        model.fit(mat_train_user_features)
        #user_item_data2 = model2.fit(mat_train_user_features)

        print("Training BPR-MF")
        model2.fit(mat_train_user_features)

        i = 0
        recommendations = pd.DataFrame()


        previoususer = -1
        recommendationsrows = pd.DataFrame(columns=['item', 'score', 'user', 'rank', 'Algorithm'])
        recommendationsrows.to_csv(filepath_write, index=False)

        print("Creating Recommendation List for Recommendation Evaluation for ALS!")
        for useri, itemi in zip(*mat_test_file_features.nonzero()):
            if (useri > previoususer):
                previoususer = useri
                recommendationsrows = pd.DataFrame()
                item, score = model.recommend(useri, mat_test_file_features[useri, :],
                                              N=500)  # indexing from csr matrix - change to features
                recommendationsrows.insert(0, "item", item)
                recommendationsrows.insert(1, "score", score)
                recommendationsrows.insert(2, "user", useri)
                recommendationsrows.insert(3, "rank", np.arange(1, item.shape[0] + 1))
                recommendationsrows["Algorithm"] = "AlternateLeastSquares"

                candidatelist = recommendationsrows.iloc[:cfg.recommendationlistsize]
                candidatelist.to_csv(filepath_write, index=False, mode='a', header=False)
                i = i + 1
        i = 0
        previoususer = -1
        print("Creating Recommendation List for Recommendation Evaluation for BPR-MF!")

        for useri, itemi in zip(*mat_test_file_features.nonzero()):
            if (useri > previoususer):
                previoususer = useri
                recommendationsrows = pd.DataFrame()
                item, score = model2.recommend(useri, mat_test_file_features[useri, :],
                                              N=500)  # indexing from csr matrix - change to features
                recommendationsrows.insert(0, "item", item)
                recommendationsrows.insert(1, "score", score)
                recommendationsrows.insert(2, "user", useri)
                recommendationsrows.insert(3, "rank", np.arange(1, item.shape[0] + 1))
                recommendationsrows["Algorithm"] = "BPR"
                candidatelist = recommendationsrows.iloc[:cfg.recommendationlistsize]
                candidatelist.to_csv(filepath_write, index=False, mode='a', header=False)

                i = i + 1

        end = dt.datetime.now()

        print("Time of end: " + str(end))
        print("Duration: " + str(end - start))



def main():
    run()

if __name__ == "__main__":
    main()