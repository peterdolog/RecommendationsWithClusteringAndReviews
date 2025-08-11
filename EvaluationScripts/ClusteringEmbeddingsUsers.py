#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#runs a clustering on the user embeddings chosen
#outputs concatenated and dotted centroid of each cluster to its neighbors and saves it to new embeddings

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import config as cfg

def run():


    if cfg.withreviewsandclusters:
        for f in range(0, cfg.folds):
            filepath_read_users = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                              + str(f) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row_withconcatreviewemb.csv"
            users = pd.read_csv(filepath_read_users, header=None)
            # dimmensionality needed for write files
            dimmensionality = len(users.columns)
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                filepath_write_users_with_clusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                                                   + "row_" + str(c) + "clusters_withconcatreviewemb_withdotclusters.csv"
                filepath_write_users_with_clusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                 "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                 str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                 "row_" + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"








                userids = users.iloc[:,0]
                print(userids)

                unorm = np.linalg.norm(users.iloc[:, 1:dimmensionality])
                users_normalized = users.iloc[:, 1:dimmensionality]/unorm
                users_normalized.columns = np.arange(len(users_normalized.columns))
                print("usersnormalized")
                print(users_normalized)

                kmeans = KMeans(n_clusters=c, random_state=0)

                y = kmeans.fit_predict(users.iloc[:, 1:dimmensionality])

                users['Cluster'] = y
                print("usersaftercluster:")
                print(users)

                dfcenters = pd.DataFrame(kmeans.cluster_centers_)
                dfclusters = pd.DataFrame(kmeans.labels_)

                dfcenters['Cluster'] = dfcenters.index
                print("cluster centres:")
                print(dfcenters)


                concatclusters = pd.merge(users, dfcenters, how="inner", on="Cluster")

                concatclusters = concatclusters.drop('Cluster', axis=1)
                concatclusters.to_csv(filepath_write_users_with_clusters_concat, index=False, header=False)
                print("after cluster concatenations:")
                print(concatclusters)


                #dot product of cluster and node representation


                dfcenters = dfcenters.drop('Cluster',axis=1)

                dfcentresnorm = np.linalg.norm(dfcenters)

                dfcentres_normalized = dfcenters/dfcentresnorm


                dfcentres_normalizedT = dfcentres_normalized.transpose()
                print("users normalized for dot product:")
                print(users_normalized)
                print("centroids normalized for dot:")
                print(dfcentres_normalizedT)


                dotclusters = users_normalized.dot(dfcentres_normalizedT)

                dotclusters.insert(0, 'userid', userids)
                print("clusters dotted to users:")
                print(dotclusters)
                dotclusters.to_csv(filepath_write_users_with_clusters_dot, index=False, header=False)
    elif cfg.withclustersandreviews:
        for f in range(0, cfg.folds):
            filepath_read_users = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/userEmb_" + cfg.whichratings + "_train" \
                              + str(f) + "_seed" + str(cfg.seed) + "filtered" + str(cfg.ratingclass) + "row.csv"
            users = pd.read_csv(filepath_read_users, header=None)
            # dimmensionality needed for write files
            dimmensionality = len(users.columns)
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                filepath_write_users_with_clusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                   "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                   str(cfg.seed) + "filtered" + str(cfg.ratingclass) \
                                                   + "row_" + str(c) + "clusters_withdotclusters.csv"
                filepath_write_users_with_clusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                 "/userEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                 str(cfg.seed) + "filtered" + str(cfg.ratingclass) + \
                                                 "row_" + str(c) + "clusters_withconcatclusters.csv"





                userids = users.iloc[:,0]
                print(userids)

                unorm = np.linalg.norm(users.iloc[:, 1:dimmensionality])
                users_normalized = users.iloc[:, 1:dimmensionality]/unorm
                users_normalized.columns = np.arange(len(users_normalized.columns))
                print("usersnormalized")
                print(users_normalized)

                kmeans = KMeans(n_clusters=c, random_state=0)

                y = kmeans.fit_predict(users.iloc[:, 1:dimmensionality])

                users['Cluster'] = y
                print("usersaftercluster:")
                print(users)

                dfcenters = pd.DataFrame(kmeans.cluster_centers_)
                dfclusters = pd.DataFrame(kmeans.labels_)

                dfcenters['Cluster'] = dfcenters.index
                print("cluster centres:")
                print(dfcenters)


                concatclusters = pd.merge(users, dfcenters, how="inner", on="Cluster")

                concatclusters = concatclusters.drop('Cluster', axis=1)
                concatclusters.to_csv(filepath_write_users_with_clusters_concat, index=False, header=False)
                print("after cluster concatenations:")
                print(concatclusters)

                #dot product of cluster and node representation


                dfcenters = dfcenters.drop('Cluster',axis=1)

                dfcentresnorm = np.linalg.norm(dfcenters)

                dfcentres_normalized = dfcenters/dfcentresnorm





                dfcentres_normalizedT = dfcentres_normalized.transpose()
                print("users normalized for dot product:")
                print(users_normalized)
                print("centroids normalized for dot:")
                print(dfcentres_normalizedT)


                dotclusters = users_normalized.dot(dfcentres_normalizedT)

                dotclusters.insert(0, 'userid', userids)
                print("clusters dotted to users:")
                print(dotclusters)
                dotclusters.to_csv(filepath_write_users_with_clusters_dot, index=False, header=False)

def main():
    run()

if __name__ == "__main__":
    main()