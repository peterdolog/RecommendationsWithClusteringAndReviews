#Peter Dolog, Aalborg University, dolog@cs.aau.dk
#runs a clustering on the items embeddings chosen
#outputs concatenated and dotted centroid of each cluster to its neighbors and saves it to new embeddings


import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import config as cfg


def run():


    if cfg.withreviewsandclusters:
        for f in range(0, cfg.folds):
            filepath_read_items = cfg.WithReviwesEmbeddingsDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                              + str(f) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + "_withconcatreviewemb.csv"
            ##reading
            items = pd.read_csv(filepath_read_items, header=None)
            ## to know dimmensionality for write files we need to before read file
            dimmensionality = len(items.columns)
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                filepath_write_filtermovies_withclusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                        "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                        str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                        "_" + str(c) + "clusters_withconcatreviewemb_withdotclusters.csv"
                filepath_write_filtermovies_withclusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                           "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                           str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                           "_" + str(c) + "clusters_withconcatreviewemb_withconcatclusters.csv"




                itemids = items.iloc[:,0]

                unorm = np.linalg.norm(items.iloc[:, 1:dimmensionality])
                items_normalized = items.iloc[:, 1:dimmensionality]/unorm
                items_normalized.columns = np.arange(len(items_normalized.columns))
                print("items normalized")
                print(items_normalized)


                kmeans = KMeans(n_clusters=c, random_state=0)

                y = kmeans.fit_predict(items.iloc[:, 1:dimmensionality])

                items['Cluster'] = y
                print("items after clustering")
                print(items)

                dfcenters = pd.DataFrame(kmeans.cluster_centers_)
                dfclusters = pd.DataFrame(kmeans.labels_)

                dfcenters['Cluster'] = dfcenters.index
                print("centroids:")
                print(dfcenters)


                concatclusters = pd.merge(items, dfcenters, how="inner", on="Cluster")

                concatclusters = concatclusters.drop('Cluster', axis=1)
                concatclusters.to_csv(filepath_write_filtermovies_withclusters_concat, index=False, header=False)
                print("clusters concatenated to items:")
                print(concatclusters)

                dfcenters = dfcenters.drop('Cluster',axis=1)

                dfcentresnorm = np.linalg.norm(dfcenters)

                dfcentres_normalized = dfcenters/dfcentresnorm


                dfcentres_normalizedT = dfcentres_normalized.transpose()
                print("items normalized transponded for dot")
                print(items_normalized)
                print("centroids normalized transponded")
                print(dfcentres_normalizedT)


                dotclusters = items_normalized.dot(dfcentres_normalizedT)
                dotclusters.insert(0, 'itemid', itemids)
                print("dotclusters:")
                print(dotclusters)
                dotclusters.to_csv(filepath_write_filtermovies_withclusters_dot, index=False, header=False)


    elif cfg.withclustersandreviews:
        for f in range(0, cfg.folds):
            filepath_read_items = cfg.SplitToClassDir + "seed" + str(cfg.seed) + "/movieEmb_" + cfg.whichratings + "_train" \
                              + str(f) + "_seed" + str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + ".csv"
            ##reading
            items = pd.read_csv(filepath_read_items, header=None)
            ## to know dimmensionality for write files we need to before read file
            dimmensionality = len(items.columns)
            for c in range(cfg.minclusters, cfg.maxclusters+1):
                filepath_write_filtermovies_withclusters_dot = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                        "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                        str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                        "_" + str(c) + "clusters_withdotclusters.csv"
                filepath_write_filtermovies_withclusters_concat = cfg.EmbeddingsFolderWithClusters + "seed" + str(cfg.seed) + \
                                                           "/movieEmb_" + cfg.whichratings + "_train" + str(f) + "_seed" + \
                                                           str(cfg.seed) + "filteredclass" + str(cfg.ratingclass) + \
                                                           "_" + str(c) + "clusters_withconcatclusters.csv"




                itemids = items.iloc[:,0]

                unorm = np.linalg.norm(items.iloc[:, 1:dimmensionality])
                items_normalized = items.iloc[:, 1:dimmensionality]/unorm
                items_normalized.columns = np.arange(len(items_normalized.columns))
                print("items normalized")
                print(items_normalized)


                kmeans = KMeans(n_clusters=c, random_state=0)

                y = kmeans.fit_predict(items.iloc[:, 1:dimmensionality])

                items['Cluster'] = y
                print("items after clustering")
                print(items)

                dfcenters = pd.DataFrame(kmeans.cluster_centers_)
                dfclusters = pd.DataFrame(kmeans.labels_)

                dfcenters['Cluster'] = dfcenters.index
                print("centroids:")
                print(dfcenters)

                concatclusters = pd.merge(items, dfcenters, how="inner", on="Cluster")

                concatclusters = concatclusters.drop('Cluster', axis=1)
                concatclusters.to_csv(filepath_write_filtermovies_withclusters_concat, index=False, header=False)
                print("clusters concatenated to items:")
                print(concatclusters)

                dfcenters = dfcenters.drop('Cluster',axis=1)

                dfcentresnorm = np.linalg.norm(dfcenters)

                dfcentres_normalized = dfcenters/dfcentresnorm




                dfcentres_normalizedT = dfcentres_normalized.transpose()
                print("items normalized transponded for dot")
                print(items_normalized)
                print("centroids normalized transponded")
                print(dfcentres_normalizedT)


                dotclusters = items_normalized.dot(dfcentres_normalizedT)
                dotclusters.insert(0, 'itemid', itemids)
                print("dotclusters:")
                print(dotclusters)
                dotclusters.to_csv(filepath_write_filtermovies_withclusters_dot, index=False, header=False)



def main():
    run()

if __name__ == "__main__":
    main()